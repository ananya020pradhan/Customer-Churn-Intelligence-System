import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def get_feature_names(model):
    preprocessor = model.named_steps["preprocessor"]

    numeric_features = preprocessor.transformers_[0][2]

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    categorical_features = preprocessor.transformers_[1][2]
    encoded_features = encoder.get_feature_names_out(categorical_features)

    return list(numeric_features) + list(encoded_features)


def evaluate_model(model, X_test, y_test, threshold=0.40):
    os.makedirs("outputs", exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nModel Evaluation")
    print("-" * 50)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": threshold
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=300)
    plt.close()

    # PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.2f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pr_curve.png", dpi=300)
    plt.close()

    # Feature Importance
    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model)
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(15)

    importance_df.to_csv("outputs/feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300)
    plt.close()

    joblib.dump({
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred
    }, "outputs/evaluation_data.pkl")

    return metrics