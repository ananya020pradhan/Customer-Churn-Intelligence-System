from src.data_generator import generate_customer_data
from src.train_model import train_models
from src.evaluate import evaluate_model


def main():
    print("Starting Customer Churn Prediction Project...")

    generate_customer_data(rows=1500)

    model, X_test, y_test = train_models()

    evaluate_model(model, X_test, y_test, threshold=0.40)

    print("\nProject completed successfully.")
    print("Check outputs folder for ROC, PR curve, confusion matrix, and feature importance.")


if __name__ == "__main__":
    main()