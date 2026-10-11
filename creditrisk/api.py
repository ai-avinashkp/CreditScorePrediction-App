# NB: no `from __future__ import annotations` here: FastAPI must see the real
# (dynamically built) Applicant class in the /predict signature, not a string.
import json
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model

from .audit import AuditLog
from .auth import AuthError, AuthService, InvalidCredentials, LockedOut, UserExists
from .config import Settings
from .predictor import Predictor

COOKIE = "access_token"


class Credentials(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


def _applicant_model(schema: dict[str, Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, spec in schema.items():
        if spec["kind"] == "numeric":
            # Headroom beyond the training range (the model clamps via its outer bins);
            # negatives only where the data has them (e.g. overpaid bureau balances).
            lo = spec["min"] * 10 if spec["min"] < 0 else 0
            fields[name] = (float, Field(ge=lo, le=max(spec["max"] * 10, 120)))
        elif spec["kind"] == "flag":
            fields[name] = (bool, ...)
        else:
            fields[name] = (Literal[tuple(spec["options"])], ...)  # type: ignore[valid-type]
    fields["sample_id"] = (Optional[int], Field(default=None, ge=1))
    return create_model("Applicant", __config__=ConfigDict(extra="forbid"), **fields)


def _load_samples(path, schema: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cats = [n for n, s in schema.items() if s["kind"] != "numeric"]
    return pd.read_csv(path, dtype={c: str for c in cats})


def _sample_applicant(row: pd.Series, schema: dict[str, Any]) -> dict[str, Any]:
    """CSV row -> API-shaped applicant (flags as bools, numbers as plain Python numbers)."""
    out = {}
    for name, spec in schema.items():
        v = row[name]
        if spec["kind"] == "flag":
            out[name] = str(v) == "1"
        elif spec["kind"] == "numeric":
            out[name] = v.item() if hasattr(v, "item") else v
        else:
            out[name] = str(v)
    return out


def create_app(settings: Settings | None = None, bcrypt_rounds: int = 12) -> FastAPI:
    settings = settings or Settings()
    predictor = Predictor(settings)
    auth = AuthService(settings, bcrypt_rounds=bcrypt_rounds)
    audit = AuditLog(settings.audit_log)
    Applicant = _applicant_model(predictor.schema)

    samples = _load_samples(settings.samples_path, predictor.schema)
    metrics = json.loads(settings.metrics_path.read_text()) if settings.metrics_path.exists() else {}

    app = FastAPI(title="Credit Risk Scoring Engine", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/api/docs"):
            resp.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
        if request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    def current_user(request: Request) -> str:
        token = request.cookies.get(COOKIE)
        header = request.headers.get("authorization", "")
        if not token and header.lower().startswith("bearer "):
            token = header[7:]
        if not token:
            raise HTTPException(401, "Not authenticated")
        try:
            return auth.verify_token(token)
        except InvalidCredentials as e:
            raise HTTPException(401, str(e)) from None

    def set_session(resp: Response, username: str) -> None:
        resp.set_cookie(
            COOKIE, auth.issue_token(username), httponly=True, samesite="strict",
            secure=settings.cookie_secure, max_age=settings.token_ttl_minutes * 60, path="/",
        )

    # ---- auth --------------------------------------------------------------
    @app.post("/api/auth/register", status_code=201)
    def register(body: Credentials, response: Response):
        try:
            user = auth.register(body.username, body.password)
        except UserExists as e:
            raise HTTPException(409, str(e)) from None
        except AuthError as e:
            raise HTTPException(422, str(e)) from None
        set_session(response, user)
        return {"username": user}

    @app.post("/api/auth/login")
    def login(body: Credentials, response: Response):
        try:
            user = auth.authenticate(body.username, body.password)
        except LockedOut as e:
            raise HTTPException(429, f"Too many attempts. Retry in {e.retry_after}s",
                                headers={"Retry-After": str(e.retry_after)}) from None
        except InvalidCredentials as e:
            raise HTTPException(401, str(e)) from None
        set_session(response, user)
        return {"username": user}

    @app.post("/api/auth/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: str = Depends(current_user)):
        return {"username": user}

    # ---- scoring -----------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_version": predictor.version}

    @app.get("/api/schema")
    def schema(_: str = Depends(current_user)):
        return {"features": predictor.features, "schema": predictor.schema}

    @app.get("/api/model")
    def model_info(_: str = Depends(current_user)):
        return {**metrics, "thresholds": {"approve_below": predictor.approve_below,
                                          "decline_above": predictor.decline_above}}

    @app.get("/api/samples")
    def list_samples(_: str = Depends(current_user)):
        return {"samples": [
            {"sample_id": int(r["sample_id"]), "actual_default": int(r["actual_default"]),
             "disbursed_amount": int(r["disbursed_amount"]), "ltv": float(r["ltv"]),
             "age": int(r["age"]), "bureau_band": str(r["bureau_band"])}
            for _, r in samples.iterrows()
        ]}

    def find_sample(sample_id: int) -> pd.Series | None:
        row = samples[samples["sample_id"] == sample_id] if not samples.empty else samples
        return None if row.empty else row.iloc[0]

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: int, _: str = Depends(current_user)):
        r = find_sample(sample_id)
        if r is None:
            raise HTTPException(404, "Sample not found")
        return {"sample_id": sample_id, "actual_default": int(r["actual_default"]),
                "applicant": _sample_applicant(r, predictor.schema)}

    @app.post("/api/predict")
    def predict(applicant: Applicant, user: str = Depends(current_user)):  # type: ignore[valid-type]
        data = applicant.model_dump()
        sample_id = data.pop("sample_id", None)
        result = predictor.predict(data)
        result["request_id"] = audit.record(user=user, inputs=data, result=result, sample_id=sample_id)
        if sample_id is not None and (r := find_sample(sample_id)) is not None:
            result["actual_default"] = int(r["actual_default"])
        return result

    # ---- UI ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(settings.web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
    return app
