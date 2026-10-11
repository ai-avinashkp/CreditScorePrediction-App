import math

import pandas as pd
import pytest

from creditrisk.predictor import Predictor, pd_to_score


@pytest.fixture(scope="module")
def predictor(base_settings):
    return Predictor(base_settings)


@pytest.fixture(scope="module")
def samples(base_settings):
    return pd.read_csv(base_settings.samples_path)


def _rec(predictor, row):
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in predictor.features}


def test_score_is_monotonic_in_pd():
    assert pd_to_score(0.05) > pd_to_score(0.3) > pd_to_score(0.8)
    assert 300 <= pd_to_score(0.999999) <= pd_to_score(1e-9) <= 850


def test_prediction_is_deterministic_and_in_range(predictor, samples):
    rec = _rec(predictor, samples.iloc[0])
    a, b = predictor.predict(rec), predictor.predict(rec)
    assert a == b
    assert 0 <= a["probability_of_default"] <= 1
    assert a["decision"] in {"APPROVE", "REVIEW", "DECLINE"}


def test_shap_contributions_sum_to_model_log_odds(predictor, samples):
    for i in range(5):
        res = predictor.predict(_rec(predictor, samples.iloc[i]))
        total = res["base_log_odds"] + sum(f["contribution"] for f in res["factors"])
        p = 1 / (1 + math.exp(-total))
        assert p == pytest.approx(res["probability_of_default"], abs=2e-3)


def test_factors_sorted_by_magnitude_and_excluded_feature_absent(predictor, samples):
    res = predictor.predict(_rec(predictor, samples.iloc[3]))
    mags = [abs(f["contribution"]) for f in res["factors"]]
    assert mags == sorted(mags, reverse=True)
    assert "Current_pincode_ID" not in {f["feature"] for f in res["factors"]}


def test_holdout_discrimination_is_better_than_chance(predictor, samples):
    from sklearn.metrics import roc_auc_score

    pds = [predictor.predict(_rec(predictor, r))["probability_of_default"] for _, r in samples.iterrows()]
    # 500 out-of-time loans; full OOT AUC is ~0.65, so allow for sampling noise.
    assert roc_auc_score(samples["actual_default"], pds) > 0.58


def test_model_is_warm_after_construction(predictor):
    import time

    # The constructor runs a warm-up prediction, so a fresh applicant scores in milliseconds
    # (cold, LightGBM's first predict takes ~2 s). Generous bound to stay robust on slow CI.
    rec = {**predictor.default_record(), "age": 41, "ltv": 70.0}
    t = time.perf_counter()
    predictor.predict(rec)
    assert time.perf_counter() - t < 0.5


def test_decision_bands_follow_thresholds(predictor):
    assert predictor.decision(predictor.approve_below - 1e-6) == "APPROVE"
    assert predictor.decision((predictor.approve_below + predictor.decline_above) / 2) == "REVIEW"
    assert predictor.decision(predictor.decline_above + 1e-6) == "DECLINE"
