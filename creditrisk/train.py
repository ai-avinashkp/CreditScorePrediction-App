"""Train the model: `python -m creditrisk.train [--force]`.

Skipped when an artifact already exists, so an existing model is reused as-is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Settings
from .woe import WoETransformer

TARGET = "default"
# Marital/sex-coded attribute: excluded to avoid discriminatory credit decisions.
EXCLUDED = ["personal_status"]
SEED = 42

LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=6, max_depth=4,
    min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, random_state=SEED, verbose=-1,
)


def load_raw(settings: Settings) -> pd.DataFrame:
    if settings.raw_data_path.exists():
        return pd.read_csv(settings.raw_data_path)
    from sklearn.datasets import fetch_openml

    frame = fetch_openml("credit-g", version=1, as_frame=True).frame
    frame[TARGET] = (frame.pop("class") == "bad").astype(int)
    for c in frame.select_dtypes("category"):
        frame[c] = frame[c].astype(str)
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(settings.raw_data_path, index=False)
    return frame


def ks_stat(y, p) -> float:
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    cg = (1 - d["y"]).cumsum() / (1 - d["y"]).sum()
    cb = d["y"].cumsum() / d["y"].sum()
    return float((cb - cg).abs().max())


def build_schema(X: pd.DataFrame) -> dict:
    schema = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            schema[c] = {"type": "numeric", "min": int(X[c].min()), "max": int(X[c].max()),
                         "median": float(X[c].median())}
        else:
            schema[c] = {"type": "categorical", "options": sorted(X[c].astype(str).unique())}
    return schema


def train(settings: Settings, force: bool = False) -> dict:
    if settings.model_path.exists() and not force:
        print(f"Model exists at {settings.model_path}; reusing (use --force to retrain).")
        return json.loads(settings.metrics_path.read_text()) if settings.metrics_path.exists() else {}

    df = load_raw(settings)
    X = df.drop(columns=[TARGET] + EXCLUDED)
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Cross-validated AUC with WoE fitted inside each fold (no leakage).
    cv_auc = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr):
        w = WoETransformer().fit(X_tr.iloc[tr], y_tr.iloc[tr])
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(w.transform(X_tr.iloc[tr]), y_tr.iloc[tr])
        cv_auc.append(roc_auc_score(y_tr.iloc[va], m.predict_proba(w.transform(X_tr.iloc[va]))[:, 1]))

    woe = WoETransformer().fit(X_tr, y_tr)
    model = lgb.LGBMClassifier(**LGB_PARAMS).fit(woe.transform(X_tr), y_tr)
    p_te = model.predict_proba(woe.transform(X_te))[:, 1]
    auc = roc_auc_score(y_te, p_te)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bundle = {
        "woe": woe, "model": model, "features": list(X.columns), "schema": build_schema(X),
        "version": version, "base_rate": float(y_tr.mean()),
    }
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, settings.model_path)

    metrics = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "German Credit (OpenML credit-g v1)",
        "n_train": len(X_tr), "n_test": len(X_te),
        "base_default_rate": round(float(y_tr.mean()), 4),
        "test_auc": round(float(auc), 4),
        "test_gini": round(float(2 * auc - 1), 4),
        "test_ks": round(ks_stat(y_te.to_numpy(), p_te), 4),
        "cv_auc_mean": round(float(np.mean(cv_auc)), 4),
        "cv_auc_std": round(float(np.std(cv_auc)), 4),
        "excluded_features": EXCLUDED,
        "top_iv": [{"feature": r.feature, "iv": round(r.iv, 4)} for r in woe.iv_summary().head(8).itertuples()],
        "data_sha256": hashlib.sha256(settings.raw_data_path.read_bytes()).hexdigest()[:16],
    }
    settings.metrics_path.write_text(json.dumps(metrics, indent=2))

    samples = X_te.copy()
    samples.insert(0, "sample_id", range(1, len(samples) + 1))
    samples["actual_default"] = y_te.to_numpy()
    samples.to_csv(settings.samples_path, index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    train(Settings(secret_key="train-only"), force=ap.parse_args().force)
