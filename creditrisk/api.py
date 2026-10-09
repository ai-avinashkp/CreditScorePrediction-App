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
        if spec["type"] == "numeric":
            # Allow some headroom beyond the training range; the model clamps via bins.
            fields[name] = (float, Field(ge=0, le=max(spec["max"] * 10, 120)))
        else:
            fields[name] = (Literal[tuple(spec["options"])], ...)  # type: ignore[valid-type]
    fields["sample_id"] = (Optional[int], Field(default=None, ge=1))
    return create_model("Applicant", __config__=ConfigDict(extra="forbid"), **fields)


def create_app(settings: Settings | None = None, bcrypt_rounds: int = 12) -> FastAPI:
    settings = settings or Settings()
    predictor = Predictor(settings)
    auth = AuthService(settings, bcrypt_rounds=bcrypt_rounds)
    audit = AuditLog(settings.audit_log)
    Applicant = _applicant_model(predictor.schema)

    samples = pd.read_csv(settings.samples_path) if settings.samples_path.exists() else pd.DataFrame()
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
        # Model/API/audit amounts stay in DM; the UI converts for display using this rate.
        return {"features": predictor.features, "schema": predictor.schema,
                "currency": {"code": "INR", "symbol": "₹", "per_dm": settings.inr_per_dm}}

    @app.get("/api/model")
    def model_info(_: str = Depends(current_user)):
        return {**metrics, "thresholds": {"approve_below": settings.approve_below,
                                          "decline_above": settings.decline_above}}

    @app.get("/api/samples")
    def list_samples(_: str = Depends(current_user)):
        return {"samples": [
            {"sample_id": int(r["sample_id"]), "actual_default": int(r["actual_default"]),
             "purpose": str(r["purpose"]), "credit_amount": int(r["credit_amount"]),  # DM; UI converts
             "duration": int(r["duration"]), "age": int(r["age"]),
             "label": f"#{int(r['sample_id'])} · {r['purpose']} · {int(r['duration'])}m · age {int(r['age'])}"}
            for _, r in samples.iterrows()
        ]}

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: int, _: str = Depends(current_user)):
        row = samples[samples["sample_id"] == sample_id]
        if row.empty:
            raise HTTPException(404, "Sample not found")
        rec = row.iloc[0].to_dict()
        return {"sample_id": sample_id, "actual_default": int(rec.pop("actual_default")),
                "applicant": {k: (v.item() if hasattr(v, "item") else v)
                              for k, v in rec.items() if k in predictor.features}}

    @app.post("/api/predict")
    def predict(applicant: Applicant, user: str = Depends(current_user)):  # type: ignore[valid-type]
        data = applicant.model_dump()
        sample_id = data.pop("sample_id", None)
        result = predictor.predict(data)
        result["request_id"] = audit.record(user=user, inputs=data, result=result, sample_id=sample_id)
        if sample_id is not None and not samples.empty:
            row = samples[samples["sample_id"] == sample_id]
            if not row.empty:
                result["actual_default"] = int(row.iloc[0]["actual_default"])
        return result

    # ---- UI ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(settings.web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
    return app


def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name: str):  # lazy `app` so importing the module doesn't need a trained model
    if name == "app":
        global app
        app = _default_app()
        return app
    raise AttributeError(name)
