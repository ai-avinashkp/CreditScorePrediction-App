"""Append-only JSONL audit trail of every scoring decision."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, user: str, inputs: dict[str, Any], result: dict[str, Any],
               sample_id: int | None = None) -> str:
        request_id = uuid.uuid4().hex
        entry = {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "sample_id": sample_id,
            "model_version": result["model_version"],
            "inputs": inputs,
            "probability_of_default": result["probability_of_default"],
            "score": result["score"],
            "decision": result["decision"],
            "shap": {f["feature"]: f["contribution"] for f in result["factors"]},
        }
        line = json.dumps(entry, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return request_id
