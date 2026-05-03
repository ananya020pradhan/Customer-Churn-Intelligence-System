import joblib
import pandas as pd


def add_single_customer_features(df):
    df = df.copy()

    df["avg_charge_per_month"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["usage_per_login"] = df["usage_hours"] / (df["login_count"] + 1)
    df["support_risk"] = df["support_tickets"] * 2 + df["payment_delay_days"]
    df["engagement_score"] = df["usage_hours"] * df["login_count"]
    df["support_ticket_rate"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["delay_per_charge"] = df["payment_delay_days"] / (df["monthly_charges"] + 1)
    df["value_gap"] = df["monthly_charges"] / (df["usage_hours"] + 1)

    return df


def risk_label(probability):
    if probability >= 0.75:
        return "Critical Risk"
    elif probability >= 0.55:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def recommended_action(probability):
    if probability >= 0.75:
        return "Urgent retention call, personalized discount, and success manager intervention."
    elif probability >= 0.55:
        return "Priority support callback and loyalty offer."
    elif probability >= 0.30:
        return "Engagement campaign, product education, and limited-time offer."
    return "Maintain normal engagement and satisfaction check."


def predict_customer(customer_data):
    model = joblib.load("models/churn_model.pkl")

    df = pd.DataFrame([customer_data])
    df = add_single_customer_features(df)

    probability = model.predict_proba(df)[0][1]
    prediction = 1 if probability >= 0.40 else 0

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4),
        "risk_level": risk_label(probability),
        "recommended_action": recommended_action(probability)
    }