"""Train the model: `python -m creditrisk.train [--force]`.

Data: L&T vehicle loan default (Kaggle `mamtadhaker/lt-vehicle-loan-default-prediction`),
expected at data/lt_vehicle_loan/train.csv. Skipped when an artifact already exists.

Model: LightGBM on Weight of Evidence features (every input WoE-binned).
Validation is out-of-time: fit on Aug-Sep 2018 disbursals, test on Oct 2018, which is how
the model would actually be used (scoring future applicants). Two baselines are trained on
the same split so the LightGBM lift is measured, not assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Settings
from .features import CATALOGUE, CATEGORICAL, EXCLUDED, FEATURES, TARGET, engineer
from .woe import WoETransformer

SEED = 42
OOT_START = "2018-10-01"
N_SAMPLES = 500
# Decision policy on training-period PDs: approve the safest 50%, decline the riskiest 15%.
APPROVE_PCT, DECLINE_PCT = 50, 85

LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=SEED, verbose=-1,
)
DEPLOYED = "LightGBM on WoE (deployed)"


def ks_stat(y, p) -> float:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p")
    cg = (1 - d["y"]).cumsum() / (1 - d["y"]).sum()
    cb = d["y"].cumsum() / d["y"].sum()
    return float((cb - cg).abs().max())


def evaluate(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(ks_stat(y, p), 4)}


def build_schema(X: pd.DataFrame) -> dict:
    schema = {}
    for c in FEATURES:
        meta = dict(CATALOGUE[c])
        if meta["kind"] == "numeric":
            meta.update(min=float(X[c].min()), max=float(X[c].max()), median=float(X[c].median()))
        else:
            counts = X[c].value_counts()
            meta["options"] = sorted(counts.index, key=lambda v: (-counts[v], v)) if meta["kind"] == "categorical" else ["0", "1"]
            meta["default"] = str(counts.index[0])
        schema[c] = meta
    return schema


def _quiet_fit(model, *args, **kw):
    with warnings.catch_warnings():  # lightgbm>=4.6 renames eval_set; keep it for older versions
        warnings.simplefilter("ignore")
        return model.fit(*args, **kw)


def fit_lgbm(Xw: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Early-stopped on a random 10% slice of the training period (never on the OOT month)."""
    X_fit, X_es, y_fit, y_es = train_test_split(Xw, y, test_size=0.1, stratify=y, random_state=SEED)
    model = _quiet_fit(lgb.LGBMClassifier(**LGB_PARAMS), X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
                       callbacks=[lgb.early_stopping(100, verbose=False)])
    best = model.best_iteration_ or LGB_PARAMS["n_estimators"]
    return lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": best}).fit(Xw, y)


def train(settings: Settings, force: bool = False) -> dict:
    if settings.model_path.exists() and not force:
        print(f"Model exists at {settings.model_path}; reusing (use --force to retrain).")
        return json.loads(settings.metrics_path.read_text()) if settings.metrics_path.exists() else {}
    if not settings.raw_data_path.exists():
        raise FileNotFoundError(
            f"{settings.raw_data_path} missing. Download with:\n  kaggle datasets download "
            "mamtadhaker/lt-vehicle-loan-default-prediction -p data/lt_vehicle_loan --unzip")

    raw = pd.read_csv(settings.raw_data_path)
    df = engineer(raw)
    df[TARGET] = raw[TARGET].to_numpy()
    oot = df["disbursal_date"] >= OOT_START
    X_tr, y_tr = df.loc[~oot, FEATURES], df.loc[~oot, TARGET]
    X_te, y_te = df.loc[oot, FEATURES], df.loc[oot, TARGET]

    # Stability check: 3-fold CV inside the training period, WoE refitted per fold (no leakage).
    cv_auc = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_tr, y_tr):
        w = WoETransformer(n_bins=10).fit(X_tr.iloc[tr], y_tr.iloc[tr], CATEGORICAL)
        m = fit_lgbm(w.transform(X_tr.iloc[tr]), y_tr.iloc[tr])
        cv_auc.append(roc_auc_score(y_tr.iloc[va], m.predict_proba(w.transform(X_tr.iloc[va]))[:, 1]))

    woe = WoETransformer(n_bins=10).fit(X_tr, y_tr, CATEGORICAL)
    W_tr, W_te = woe.transform(X_tr), woe.transform(X_te)
    model = fit_lgbm(W_tr, y_tr)
    p_te = model.predict_proba(W_te)[:, 1]

    # Baselines on the identical split.
    band_only = LogisticRegression().fit(W_tr[["bureau_band"]], y_tr)
    scorecard = LogisticRegression(max_iter=2000).fit(W_tr, y_tr)
    comparison = {
        "Bureau band only (rule baseline)": evaluate(y_te, band_only.predict_proba(W_te[["bureau_band"]])[:, 1]),
        "Logistic regression scorecard (WoE)": evaluate(y_te, scorecard.predict_proba(W_te)[:, 1]),
        DEPLOYED: evaluate(y_te, p_te),
    }

    p_tr = model.predict_proba(W_tr)[:, 1]
    thresholds = {"approve_below": round(float(np.percentile(p_tr, APPROVE_PCT)), 4),
                  "decline_above": round(float(np.percentile(p_tr, DECLINE_PCT)), 4)}

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bundle = {
        "woe": woe, "model": model, "features": FEATURES, "schema": build_schema(X_tr),
        "version": version, "thresholds": thresholds,
    }
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, settings.model_path, compress=3)

    deployed = comparison[DEPLOYED]
    metrics = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "L&T Vehicle Loan Default (India, two-wheeler loans, Aug-Oct 2018)",
        "target": "Missed first EMI on due date",
        "validation": f"Out-of-time: train on disbursals before {OOT_START}, test on Oct 2018",
        "deployed_model": DEPLOYED,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "base_default_rate": round(float(y_tr.mean()), 4),
        "test_default_rate": round(float(y_te.mean()), 4),
        "test_auc": deployed["auc"], "test_gini": deployed["gini"], "test_ks": deployed["ks"],
        "cv_auc_mean": round(float(np.mean(cv_auc)), 4), "cv_auc_std": round(float(np.std(cv_auc)), 4),
        "n_trees": int(model.n_estimators),
        "baselines": comparison,
        "thresholds": thresholds,
        "excluded_features": EXCLUDED,
        "top_iv": [{"feature": r.feature, "iv": round(r.iv, 4)} for r in woe.iv_summary().head(8).itertuples()],
        "data_sha256": hashlib.sha256(settings.raw_data_path.read_bytes()).hexdigest()[:16],
    }
    settings.metrics_path.write_text(json.dumps(metrics, indent=2))

    samples = X_te.sample(N_SAMPLES, random_state=SEED).copy()
    samples["actual_default"] = y_te.loc[samples.index].to_numpy()
    samples.insert(0, "sample_id", range(1, len(samples) + 1))
    settings.samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(settings.samples_path, index=False)
    print(json.dumps({k: metrics[k] for k in ("test_auc", "test_ks", "cv_auc_mean", "n_trees", "baselines",
                                              "thresholds")}, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    train(Settings(secret_key="train-only"), force=ap.parse_args().force)
