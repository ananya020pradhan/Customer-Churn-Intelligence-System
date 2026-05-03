import os
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_data, add_features, split_data, build_preprocessor


def train_models():
    os.makedirs("models", exist_ok=True)

    df = load_data()
    df = add_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")

    print("Model trained successfully.")
    print("Model saved at models/churn_model.pkl")

    return model, X_test, y_test