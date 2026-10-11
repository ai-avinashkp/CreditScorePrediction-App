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

## Phase 6 - Switch to Indian data (L&T vehicle loans)
- [x] 6.1 Download via Kaggle CLI, profile (233k rows, 21.7% first-EMI default, Aug-Oct 2018)
- [x] 6.2 Feature catalogue + engineering (age, months, PRI+SEC aggregation); exclude pincode/employee/dealer IDs
- [x] 6.3 Out-of-time validation (train Aug-Sep, test Oct) + baselines (bureau band, LR scorecard)
- [x] 6.4 Decision cut-offs from score distribution (approve safest 50%, decline riskiest 15%)
- [x] 6.5 API + UI driven by /api/schema; native INR, DM conversion removed; KYC flags as checkboxes
- [x] 6.6 Tests updated/added (45 passing)
- [ ] 6.7 Visual check of the new form in a browser

## Phase 7 - Model improvement (LightGBM kept)
- [x] 7.1 Experiments selected on Sep validation (train Aug): WoE-only 0.659 -> raw 0.662 -> +ratios/branch volume 0.663
- [x] 7.2 24-trial random search (gain within noise, kept best config)
- [x] 7.3 Hybrid design (WoE for categoricals, raw numerics + derived ratios); Oct OOT 0.6487 -> 0.6539
- [x] 7.4 Decision: rolled back to the WoE-only LightGBM (Oct AUC 0.6495) for traceability; hybrid code removed

## Phase 8 - Refactor
- [x] 8.1 Removed hybrid/derived-feature code, unused helpers (WoE fit_transform, lazy app shim, base_rate)
- [x] 8.2 Runtime vs dev requirements split; dropped unused python-multipart
- [x] 8.3 Tests updated

## Later scope (not now)
- Dockerfile / image build and cloud deploy
- Feast feature store + S3 data versioning
- Drift monitoring, model registry
