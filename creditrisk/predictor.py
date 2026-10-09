from __future__ import annotations

import math
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .config import Settings

# Scorecard scaling: 600 points at 30:1 good:bad odds, 40 points to double the odds.
_PDO, _BASE_SCORE, _BASE_ODDS = 40.0, 600.0, 30.0
_FACTOR = _PDO / math.log(2)
_OFFSET = _BASE_SCORE - _FACTOR * math.log(_BASE_ODDS)


def pd_to_score(pd_: float) -> int:
    pd_ = min(max(pd_, 1e-6), 1 - 1e-6)
    score = _OFFSET + _FACTOR * math.log((1 - pd_) / pd_)
    return int(round(min(max(score, 300), 850)))


class Predictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        bundle = joblib.load(settings.model_path)
        self.woe = bundle["woe"]
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.schema: dict[str, Any] = bundle["schema"]
        self.version: str = bundle["version"]
        self.base_rate: float = bundle["base_rate"]
        self._explainer = shap.TreeExplainer(self.model)

    def decision(self, pd_: float) -> str:
        if pd_ < self.settings.approve_below:
            return "APPROVE"
        if pd_ > self.settings.decline_above:
            return "DECLINE"
        return "REVIEW"

    def predict(self, record: dict[str, Any], top_k: int | None = None) -> dict[str, Any]:
        X = pd.DataFrame([record], columns=self.features)
        W = self.woe.transform(X)
        pd_ = float(self.model.predict_proba(W)[0, 1])

        sv = self._explainer.shap_values(W)
        if isinstance(sv, list):  # older shap: [class0, class1]
            sv = sv[-1]
        contrib = np.asarray(sv)[0]
        base = np.atleast_1d(self._explainer.expected_value)[-1]
        bins = self.woe.bins(X).iloc[0]

        factors = [
            {
                "feature": f,
                "value": record[f],
                "bin": str(bins[f]),
                "woe": round(float(W.iloc[0][f]), 4),
                "contribution": round(float(c), 4),  # log-odds of default; + raises risk
                "direction": "increases_risk" if c > 0 else "decreases_risk",
            }
            for f, c in zip(self.features, contrib)
        ]
        factors.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        if top_k:
            factors = factors[:top_k]

        return {
            "probability_of_default": round(pd_, 4),
            "score": pd_to_score(pd_),
            "decision": self.decision(pd_),
            "base_log_odds": round(float(base), 4),
            "base_probability": round(float(1 / (1 + math.exp(-base))), 4),
            "factors": factors,
            "model_version": self.version,
        }
