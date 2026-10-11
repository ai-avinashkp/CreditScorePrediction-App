"""Feature catalogue and raw -> model feature engineering for the L&T vehicle loan data.

The catalogue is the single source of truth for what the model takes as input and how the
UI presents it (served via /api/schema), so labels never drift between backend and frontend.
"""
from __future__ import annotations

import pandas as pd

TARGET = "loan_default"  # missed the first EMI on its due date

# Deliberately NOT used (fairness / compliance), see metrics.json "excluded_features":
EXCLUDED = {
    "Current_pincode_ID": "customer pincode can act as a proxy for community or religion",
    "Employee_code_ID": "operational: identifies the staff member, not borrower risk",
    "supplier_id": "operational: identifies the dealer, not borrower risk",
    "Date.of.Birth": "replaced by derived age",
    "MobileNo_Avl_Flag": "constant in the data",
    "UniqueID": "identifier",
}

# name -> (kind, group, label, unit, hint)
# kind: numeric | categorical | flag ; unit "inr" is shown with the rupee sign
CATALOGUE: dict[str, dict] = {}


def _f(name, kind, group, label, unit=None, hint=None):
    CATALOGUE[name] = {"kind": kind, "group": group, "label": label, "unit": unit, "hint": hint}


LOAN, APPL, BUREAU = "Loan details", "Applicant & KYC", "Credit bureau"
_f("disbursed_amount", "numeric", LOAN, "Loan amount", "inr")
_f("asset_cost", "numeric", LOAN, "Vehicle cost", "inr")
_f("ltv", "numeric", LOAN, "Loan-to-value (%)", hint="Loan amount as a % of vehicle value")
_f("manufacturer_id", "categorical", LOAN, "Manufacturer", hint="Anonymised in the source data")
_f("branch_id", "categorical", LOAN, "Branch", hint="Anonymised in the source data")
_f("state_id", "categorical", LOAN, "State", hint="Anonymised in the source data")

_f("age", "numeric", APPL, "Age (years)")
_f("employment_type", "categorical", APPL, "Employment type")
_f("aadhar_flag", "flag", APPL, "Aadhaar provided")
_f("pan_flag", "flag", APPL, "PAN provided")
_f("voterid_flag", "flag", APPL, "Voter ID provided")
_f("driving_flag", "flag", APPL, "Driving licence provided")
_f("passport_flag", "flag", APPL, "Passport provided")

_f("bureau_score", "numeric", BUREAU, "Bureau score", hint="0 when no score is available")
_f("bureau_band", "categorical", BUREAU, "Bureau risk band")
_f("total_accounts", "numeric", BUREAU, "Loans ever taken", hint="Own loans plus co-applicant / guarantor loans")
_f("active_accounts", "numeric", BUREAU, "Active loans")
_f("overdue_accounts", "numeric", BUREAU, "Loans currently overdue")
_f("current_balance", "numeric", BUREAU, "Outstanding principal", "inr", "Can be negative (overpaid) in bureau data")
_f("sanctioned_amount", "numeric", BUREAU, "Total sanctioned on all loans", "inr")
_f("instalment_amount", "numeric", BUREAU, "Existing EMIs per month", "inr")
_f("new_accounts_6m", "numeric", BUREAU, "New loans in last 6 months")
_f("delinquent_accounts_6m", "numeric", BUREAU, "Loans defaulted in last 6 months")
_f("avg_account_age_months", "numeric", BUREAU, "Average loan age (months)")
_f("credit_history_months", "numeric", BUREAU, "Credit history length (months)")
_f("inquiries", "numeric", BUREAU, "Loan enquiries")

FEATURES = list(CATALOGUE)
CATEGORICAL = [n for n, s in CATALOGUE.items() if s["kind"] in ("categorical", "flag")]


def _months(s: pd.Series) -> pd.Series:
    """'1yrs 11mon' -> 23"""
    parts = s.str.extract(r"(\d+)yrs\s+(\d+)mon").astype(int)
    return parts[0] * 12 + parts[1]


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw L&T columns -> model features (same names as CATALOGUE) + disbursal date."""
    disb = pd.to_datetime(raw["DisbursalDate"], format="%d-%m-%y")
    dob = pd.to_datetime(raw["Date.of.Birth"], format="%d-%m-%y")
    dob = dob.where(dob < disb, dob - pd.DateOffset(years=100))  # 2-digit years: '01-01-64' parses as 2064

    out = pd.DataFrame({
        "disbursed_amount": raw["disbursed_amount"],
        "asset_cost": raw["asset_cost"],
        "ltv": raw["ltv"],
        "manufacturer_id": raw["manufacturer_id"].astype(str),
        "branch_id": raw["branch_id"].astype(str),
        "state_id": raw["State_ID"].astype(str),
        "age": ((disb - dob).dt.days / 365.25).round().astype(int),
        "employment_type": raw["Employment.Type"].fillna("Not provided"),
        "aadhar_flag": raw["Aadhar_flag"].astype(str),
        "pan_flag": raw["PAN_flag"].astype(str),
        "voterid_flag": raw["VoterID_flag"].astype(str),
        "driving_flag": raw["Driving_flag"].astype(str),
        "passport_flag": raw["Passport_flag"].astype(str),
        "bureau_score": raw["PERFORM_CNS.SCORE"],
        "bureau_band": raw["PERFORM_CNS.SCORE.DESCRIPTION"],
        "total_accounts": raw["PRI.NO.OF.ACCTS"] + raw["SEC.NO.OF.ACCTS"],
        "active_accounts": raw["PRI.ACTIVE.ACCTS"] + raw["SEC.ACTIVE.ACCTS"],
        "overdue_accounts": raw["PRI.OVERDUE.ACCTS"] + raw["SEC.OVERDUE.ACCTS"],
        "current_balance": raw["PRI.CURRENT.BALANCE"] + raw["SEC.CURRENT.BALANCE"],
        "sanctioned_amount": raw["PRI.SANCTIONED.AMOUNT"] + raw["SEC.SANCTIONED.AMOUNT"],
        "instalment_amount": raw["PRIMARY.INSTAL.AMT"] + raw["SEC.INSTAL.AMT"],
        "new_accounts_6m": raw["NEW.ACCTS.IN.LAST.SIX.MONTHS"],
        "delinquent_accounts_6m": raw["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"],
        "avg_account_age_months": _months(raw["AVERAGE.ACCT.AGE"]),
        "credit_history_months": _months(raw["CREDIT.HISTORY.LENGTH"]),
        "inquiries": raw["NO.OF_INQUIRIES"],
    })
    assert list(out.columns) == FEATURES
    out["disbursal_date"] = disb
    return out


def to_model_frame(records: list[dict]) -> pd.DataFrame:
    """API records -> DataFrame with the dtypes the WoE encoder was fitted on (categoricals as str)."""
    df = pd.DataFrame(records, columns=FEATURES)
    for c in CATEGORICAL:
        col = df[c]
        if CATALOGUE[c]["kind"] == "flag":
            col = col.map(lambda v: "1" if v in (True, 1, "1") else "0")
        df[c] = col.astype(str)
    return df
