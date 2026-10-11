# Credit Risk Scoring Engine

Explainable default-risk scoring for Indian two-wheeler loans: LightGBM on Weight of Evidence
(WoE) features, SHAP reason codes per decision, a FastAPI service with authentication and an
append-only audit log, and a responsive neumorphic web UI.

## Data

[L&T Vehicle Loan Default](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction)
(Indian NBFC, 233k loans disbursed Aug–Oct 2018). The target is a missed first EMI (21.7% of loans).
The raw data is not committed (Kaggle licence "other"). To download it:

```
kaggle datasets download mamtadhaker/lt-vehicle-loan-default-prediction -p data/lt_vehicle_loan --unzip
```

The trained model (`artifacts/`) and 500 hold-out loans (`data/test_samples.csv`) are committed,
so the app runs without the raw data. You only need the raw data to retrain.

## Results (out-of-time: trained on Aug–Sep, tested on 98k October loans)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Bureau band only (rule baseline) | 0.555 | 0.109 | 0.079 |
| Logistic regression scorecard on WoE | 0.635 | 0.269 | 0.194 |
| **LightGBM on WoE (deployed)** | **0.650** | **0.299** | **0.220** |

Every input is WoE-binned (10 quantile bins for numerics, one bin per category), so each SHAP
reason maps to a named bin a credit officer can check. First-EMI default is hard to predict from
application data, and this dataset is known for modest AUCs.

A hybrid variant (raw numerics, WoE categoricals and derived ratios) reached 0.654. It was not
deployed: the gain is small, and the WoE-only model keeps every input traceable to a bin.

**Excluded on purpose:** customer pincode (can proxy for community or religion), plus employee
and dealer IDs (operational, not borrower risk). See `creditrisk/features.py`.

## Run (conda env `envPyTorch`)

```
pip install -r requirements-dev.txt   # runtime deps are in requirements.txt
python -m creditrisk.train            # skipped if artifacts/model.joblib exists; --force to retrain
$env:CREDIT_SECRET_KEY = "<long random string>"   # JWT signing key; random per run if unset
python -m uvicorn creditrisk.api:create_app --factory --port 8000
python -m pytest
```

Then open http://127.0.0.1:8000.

## Layout

- `creditrisk/features.py`: feature catalogue (single source of truth for the model and UI) and feature engineering
- `creditrisk/woe.py`: WoE binning and encoding
- `creditrisk/train.py`: out-of-time training, baselines, metrics, decision cut-offs
- `creditrisk/predictor.py`: probability of default, 300–850 score, decision, SHAP factors
- `creditrisk/api.py`, `auth.py`, `audit.py`: service, JWT cookie auth with lockout, JSONL audit trail
- `web/`: UI, served at `/` (no build step)
