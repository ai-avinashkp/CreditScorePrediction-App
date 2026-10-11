from __future__ import annotations

import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    model_path: Path = ROOT / "artifacts" / "model.joblib"
    metrics_path: Path = ROOT / "artifacts" / "metrics.json"
    samples_path: Path = ROOT / "data" / "test_samples.csv"
    raw_data_path: Path = ROOT / "data" / "lt_vehicle_loan" / "train.csv"
    users_db: Path = ROOT / "data" / "users.db"
    audit_log: Path = ROOT / "data" / "audit" / "predictions.jsonl"
    web_dir: Path = ROOT / "web"

    secret_key: str = ""
    token_ttl_minutes: int = 60
    cookie_secure: bool = False  # set CREDIT_COOKIE_SECURE=1 behind HTTPS
    max_failed_logins: int = 5
    lockout_seconds: int = 60

    # PD cut-offs for the decision band. None = use the policy cut-offs stored with the model
    # at training time (approve safest 50%, decline riskiest 15%); set to override.
    approve_below: float | None = None
    decline_above: float | None = None

    def __post_init__(self) -> None:
        if not self.secret_key:
            self.secret_key = os.getenv("CREDIT_SECRET_KEY", "")
        if not self.secret_key:
            warnings.warn(
                "CREDIT_SECRET_KEY not set: using a random key, sessions reset on restart.",
                stacklevel=2,
            )
            self.secret_key = secrets.token_urlsafe(48)
        self.cookie_secure = self.cookie_secure or os.getenv("CREDIT_COOKIE_SECURE") == "1"
