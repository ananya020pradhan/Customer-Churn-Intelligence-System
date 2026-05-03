import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path="data/churn_data.csv"):
    return pd.read_csv(path)


def add_features(df):
    df = df.copy()

    df["avg_charge_per_month"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["usage_per_login"] = df["usage_hours"] / (df["login_count"] + 1)
    df["support_risk"] = df["support_tickets"] * 2 + df["payment_delay_days"]
    df["engagement_score"] = df["usage_hours"] * df["login_count"]
    df["support_ticket_rate"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["delay_per_charge"] = df["payment_delay_days"] / (df["monthly_charges"] + 1)
    df["value_gap"] = df["monthly_charges"] / (df["usage_hours"] + 1)

    return df


def split_data(df):
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor