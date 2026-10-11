import json

from tests.conftest import PASSWORD


def test_health_is_public(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_protected_endpoints_require_auth(client):
    for path in ["/api/schema", "/api/model", "/api/samples", "/api/samples/1", "/api/auth/me"]:
        assert client.get(path).status_code == 401, path
    assert client.post("/api/predict", json={}).status_code == 401


def test_register_sets_httponly_cookie_and_me_works(client):
    r = client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    assert r.status_code == 201
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/auth/me").json() == {"username": "alice"}


def test_login_logout_flow(client):
    client.post("/api/auth/register", json={"username": "bob", "password": PASSWORD})
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "bob", "password": "nope-nope1"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "bob", "password": PASSWORD}).status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_bearer_header_is_accepted(client):
    client.post("/api/auth/register", json={"username": "carl", "password": PASSWORD})
    token = client.cookies.get("access_token")
    client.cookies.clear()
    assert client.get("/api/auth/me").status_code == 401
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.json()["username"] == "carl"


def test_login_lockout_returns_429(client):
    client.post("/api/auth/register", json={"username": "dora", "password": PASSWORD})
    client.cookies.clear()
    for _ in range(3):
        client.post("/api/auth/login", json={"username": "dora", "password": "bad-pass-1"})
    r = client.post("/api/auth/login", json={"username": "dora", "password": PASSWORD})
    assert r.status_code == 429 and "retry-after" in r.headers


def test_duplicate_registration_conflict_and_weak_password(client):
    assert client.post("/api/auth/register", json={"username": "eve", "password": PASSWORD}).status_code == 201
    assert client.post("/api/auth/register", json={"username": "eve", "password": PASSWORD}).status_code == 409
    assert client.post("/api/auth/register", json={"username": "eve2", "password": "weak"}).status_code == 422


def test_schema_and_samples(authed):
    schema = authed.get("/api/schema").json()
    for excluded in ("Current_pincode_ID", "Employee_code_ID", "supplier_id"):
        assert excluded not in schema["features"]
    assert schema["schema"]["disbursed_amount"]["unit"] == "inr"
    assert schema["schema"]["aadhar_flag"]["kind"] == "flag"
    samples = authed.get("/api/samples").json()["samples"]
    assert len(samples) == 500 and {"sample_id", "actual_default", "disbursed_amount", "bureau_band"} <= samples[0].keys()
    assert authed.get("/api/samples/999999").status_code == 404


def test_predict_sample_returns_explanation_and_actual(authed, applicant):
    r = authed.post("/api/predict", json={**applicant, "sample_id": 1})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert 300 <= body["score"] <= 850
    assert {f["feature"] for f in body["factors"]} == set(applicant)
    assert body["actual_default"] in (0, 1) and body["request_id"]


def test_predict_validation(authed, applicant):
    assert isinstance(applicant["aadhar_flag"], bool) and isinstance(applicant["branch_id"], str)
    bad_cat = {**applicant, "employment_type": "Astronaut"}
    assert authed.post("/api/predict", json=bad_cat).status_code == 422
    missing = {k: v for k, v in applicant.items() if k != "age"}
    assert authed.post("/api/predict", json=missing).status_code == 422
    extra = {**applicant, "is_admin": True}
    assert authed.post("/api/predict", json=extra).status_code == 422
    neg = {**applicant, "disbursed_amount": -5}
    assert authed.post("/api/predict", json=neg).status_code == 422
    # bureau balances can legitimately be negative (overpaid), so that field accepts them
    overpaid = {**applicant, "current_balance": -1000}
    assert authed.post("/api/predict", json=overpaid).status_code == 200


def test_model_info_reports_baselines_and_thresholds(authed):
    m = authed.get("/api/model").json()
    deployed = m["baselines"][m["deployed_model"]]["auc"]
    assert deployed == m["test_auc"]
    assert all(deployed >= b["auc"] for b in m["baselines"].values())
    assert 0 < m["thresholds"]["approve_below"] < m["thresholds"]["decline_above"] < 1


def test_prediction_is_audited(authed, applicant, settings):
    rid = authed.post("/api/predict", json={**applicant, "sample_id": 1}).json()["request_id"]
    lines = settings.audit_log.read_text().strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["request_id"] == rid and entry["user"] == "alice"
    assert entry["inputs"]["age"] == applicant["age"]
    assert set(entry["shap"]) == set(applicant) and entry["model_version"]


def test_security_headers_and_ui(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/").status_code == 200
