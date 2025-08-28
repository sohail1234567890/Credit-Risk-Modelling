import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------
# Load the model and preprocessing objects
# -------------------------------------------------------------------

# Build absolute path relative to this file (avoids FileNotFound issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_data.joblib")

print("Loading model from:", MODEL_PATH)  # Debug info

# Load model data
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
cols_to_scale = model_data["cols_to_scale"]


# -------------------------------------------------------------------
# Helper to prepare the input data
# -------------------------------------------------------------------
def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    """Prepares a single-row DataFrame for model prediction."""

    # Build input dict
    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_amount / income if income > 0 else 0,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,

        # One-hot encoded categorical features
        "residence_type_Owned": 1 if residence_type == "Owned" else 0,
        "residence_type_Rented": 1 if residence_type == "Rented" else 0,
        "loan_purpose_Education": 1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home": 1 if loan_purpose == "Home" else 0,
        "loan_purpose_Personal": 1 if loan_purpose == "Personal" else 0,
        "loan_type_Unsecured": 1 if loan_type == "Unsecured" else 0,

        # Dummy values for unused but expected columns
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1,
    }

    # Build DataFrame
    df = pd.DataFrame([input_data])

    # Scale only required columns (safe reindex in case of missing cols)
    for col in cols_to_scale:
        if col not in df:
            df[col] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Keep only the features used by the model
    df = df.reindex(columns=features, fill_value=0)

    return df


# -------------------------------------------------------------------
# Prediction function
# -------------------------------------------------------------------
def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    """Runs the model prediction and returns probability, credit score, and rating."""

    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


# -------------------------------------------------------------------
# Credit score + rating logic
# -------------------------------------------------------------------
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """Calculates default probability, credit score, and rating."""

    # If model has predict_proba (safer than manual coef_)
    if hasattr(model, "predict_proba"):
        default_probability = model.predict_proba(input_df)[:, 1][0]
    else:
        # Fall back to manual logistic regression math
        x = np.dot(input_df.values, model.coef_.T) + model.intercept_
        default_probability = 1 / (1 + np.exp(-x))
        default_probability = float(default_probability.flatten()[0])

    non_default_probability = 1 - default_probability

    # Scale to credit score range [300, 900]
    credit_score = base_score + non_default_probability * scale_length
    credit_score = int(credit_score)

    # Assign rating bucket
    if credit_score < 500:
        rating = "Poor"
    elif credit_score < 650:
        rating = "Average"
    elif credit_score < 750:
        rating = "Good"
    else:
        rating = "Excellent"

    return round(default_probability, 4), credit_score, rating


# -------------------------------------------------------------------
# Quick test harness (only runs if you run this file directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    prob, score, rating = predict(
        age=30, income=500000, loan_amount=200000,
        loan_tenure_months=36, avg_dpd_per_delinquency=5,
        delinquency_ratio=20, credit_utilization_ratio=30,
        num_open_accounts=2, residence_type="Owned",
        loan_purpose="Home", loan_type="Secured"
    )
    print(f"Default probability: {prob}")
    print(f"Credit score: {score}")
    print(f"Rating: {rating}")
