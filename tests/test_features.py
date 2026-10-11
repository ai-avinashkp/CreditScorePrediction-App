import pandas as pd

from creditrisk.features import CATALOGUE, FEATURES, engineer, to_model_frame


def _raw(**over):
    row = {
        "UniqueID": 1, "disbursed_amount": 50000, "asset_cost": 60000, "ltv": 83.3, "branch_id": 67,
        "supplier_id": 1, "manufacturer_id": 45, "Current_pincode_ID": 1, "Date.of.Birth": "01-01-64",
        "Employment.Type": None, "DisbursalDate": "03-08-18", "State_ID": 6, "Employee_code_ID": 1,
        "MobileNo_Avl_Flag": 1, "Aadhar_flag": 1, "PAN_flag": 0, "VoterID_flag": 0, "Driving_flag": 0,
        "Passport_flag": 0, "PERFORM_CNS.SCORE": 700, "PERFORM_CNS.SCORE.DESCRIPTION": "C-Very Low Risk",
        "PRI.NO.OF.ACCTS": 3, "PRI.ACTIVE.ACCTS": 2, "PRI.OVERDUE.ACCTS": 1, "PRI.CURRENT.BALANCE": -500,
        "PRI.SANCTIONED.AMOUNT": 1000, "PRI.DISBURSED.AMOUNT": 1000, "SEC.NO.OF.ACCTS": 1,
        "SEC.ACTIVE.ACCTS": 1, "SEC.OVERDUE.ACCTS": 0, "SEC.CURRENT.BALANCE": 200,
        "SEC.SANCTIONED.AMOUNT": 300, "SEC.DISBURSED.AMOUNT": 300, "PRIMARY.INSTAL.AMT": 900,
        "SEC.INSTAL.AMT": 100, "NEW.ACCTS.IN.LAST.SIX.MONTHS": 0, "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": 0,
        "AVERAGE.ACCT.AGE": "1yrs 11mon", "CREDIT.HISTORY.LENGTH": "10yrs 0mon", "NO.OF_INQUIRIES": 2,
    }
    row.update(over)
    return pd.DataFrame([row])


def test_engineer_outputs_catalogue_columns_in_order():
    out = engineer(_raw())
    assert list(out.columns[: len(FEATURES)]) == FEATURES


def test_two_digit_birth_year_is_not_in_the_future():
    # '01-01-64' parses as 2064; it must be treated as 1964 -> 54.6 years at 3 Aug 2018, rounded to 55
    assert engineer(_raw())["age"].iloc[0] == 55


def test_durations_parse_to_months_and_accounts_are_summed():
    r = engineer(_raw()).iloc[0]
    assert r["avg_account_age_months"] == 23 and r["credit_history_months"] == 120
    assert r["total_accounts"] == 4 and r["active_accounts"] == 3 and r["overdue_accounts"] == 1
    assert r["current_balance"] == -300 and r["instalment_amount"] == 1000


def test_missing_employment_and_ids_become_categories():
    r = engineer(_raw()).iloc[0]
    assert r["employment_type"] == "Not provided"
    assert r["branch_id"] == "67" and r["aadhar_flag"] == "1"


def test_excluded_columns_are_not_features():
    assert not {"Current_pincode_ID", "Employee_code_ID", "supplier_id"} & set(FEATURES)


def test_to_model_frame_maps_api_flags_to_encoder_strings():
    rec = {f: 0 for f in FEATURES}
    rec.update(aadhar_flag=True, pan_flag=False, branch_id="67")
    df = to_model_frame([rec])
    assert df.loc[0, "aadhar_flag"] == "1" and df.loc[0, "pan_flag"] == "0"
    assert all(df[c].dtype == object for c, s in CATALOGUE.items() if s["kind"] != "numeric")
