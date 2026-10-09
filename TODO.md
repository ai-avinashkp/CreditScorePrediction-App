# Credit Risk Scoring Engine - TODO

Env: `envPyTorch` (miniconda). Run everything with that interpreter.

## Phase 1 - Model (skipped if `artifacts/model.joblib` already exists)
- [x] 1.1 Data: fetch German Credit (OpenML `credit-g`), cache to `data/credit_g.csv`
- [x] 1.2 WoE transformer (quantile bins for numeric, category WoE, IV table)
- [x] 1.3 Train LightGBM on WoE features, stratified hold-out, report AUC / KS / Gini
- [x] 1.4 Persist bundle (`artifacts/model.joblib`, `metrics.json`) + hold-out set (`data/test_samples.csv`)

## Phase 2 - Serving core
- [x] 2.1 Predictor: PD, score (300-850), decision band, SHAP reason codes
- [x] 2.2 Audit log (append-only JSONL: user, inputs, PD, SHAP, model version)
- [x] 2.3 Auth: SQLite users, bcrypt, JWT in HttpOnly cookie, login throttling
- [x] 2.4 FastAPI app: /api/auth/*, /api/schema, /api/samples, /api/predict, /api/model, /api/health, security headers

## Phase 3 - UI (neumorphic, responsive, no build step)
- [x] 3.1 Login / register
- [x] 3.2 Applicant form built from /api/schema + hold-out sample picker
- [x] 3.3 Result: PD gauge, score, decision, SHAP contribution bars, actual-vs-predicted for samples
- [x] 3.4 Model card panel (metrics, top IV)

## Phase 4 - Tests (36 passing)
- [x] 4.1 WoE unit tests
- [x] 4.2 Predictor tests (range, determinism, SHAP additivity)
- [x] 4.3 Auth tests (register/login/lockout/expired/tampered token)
- [x] 4.4 API tests (authz, validation, audit written, samples)

## Phase 5 - Run & verify
- [x] API smoke-tested against live uvicorn (register, sample, predict, static files)
- [ ] Visual check of the UI in a real browser (not yet done)

## Later scope (not now)
- Dockerfile / image build and cloud deploy
- Feast feature store + S3 data versioning
- Drift monitoring, model registry
