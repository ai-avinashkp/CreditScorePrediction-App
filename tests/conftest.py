import pytest
from fastapi.testclient import TestClient

from creditrisk.api import create_app
from creditrisk.config import Settings
from creditrisk.train import train

PASSWORD = "Sup3rSecret1"


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    s = Settings(secret_key="test-secret-key-test-secret-key-0123")
    train(s)  # no-op when a model already exists
    return s


@pytest.fixture()
def settings(base_settings, tmp_path) -> Settings:
    """Real model artifacts, isolated users DB and audit log per test."""
    return Settings(
        secret_key=base_settings.secret_key,
        users_db=tmp_path / "users.db",
        audit_log=tmp_path / "audit" / "predictions.jsonl",
        max_failed_logins=3,
    )


@pytest.fixture()
def client(settings) -> TestClient:
    return TestClient(create_app(settings, bcrypt_rounds=4))


@pytest.fixture()
def authed(client) -> TestClient:
    r = client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    assert r.status_code == 201
    return client


@pytest.fixture()
def applicant(authed) -> dict:
    return authed.get("/api/samples/1").json()["applicant"]
