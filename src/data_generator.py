import pandas as pd
import numpy as np
import os


def generate_customer_data(rows=1000, save_path="data/churn_data.csv"):
    np.random.seed(42)

    os.makedirs("data", exist_ok=True)

    customer_id = [f"CUST_{i}" for i in range(1, rows + 1)]

    age = np.random.randint(18, 70, rows)
    tenure_months = np.random.randint(1, 72, rows)
    monthly_charges = np.random.randint(300, 5000, rows)
    total_charges = tenure_months * monthly_charges + np.random.randint(100, 2000, rows)

    usage_hours = np.random.randint(1, 120, rows)
    login_count = np.random.randint(1, 100, rows)
    support_tickets = np.random.randint(0, 8, rows)
    payment_delay_days = np.random.randint(0, 30, rows)
    contract_type = np.random.choice(["Monthly", "Quarterly", "Yearly"], rows)
    internet_service = np.random.choice(["Fiber", "DSL", "No"], rows)
    is_autopay = np.random.choice(["Yes", "No"], rows)
    region = np.random.choice(["East", "West", "North", "South"], rows)

    churn_probability = (
        (payment_delay_days > 18).astype(int) * 0.30
    + (support_tickets > 4).astype(int) * 0.25
    + (tenure_months < 10).astype(int) * 0.20
    + (usage_hours < 25).astype(int) * 0.20
    + (contract_type == "Monthly").astype(int) * 0.20
    + (is_autopay == "No").astype(int) * 0.15
    + (monthly_charges > 3500).astype(int) * 0.15
    )

    
    churn_probability = np.clip(churn_probability , 0.05, 0.95)
    churn = np.random.binomial(1, churn_probability)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "usage_hours": usage_hours,
        "login_count": login_count,
        "support_tickets": support_tickets,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "is_autopay": is_autopay,
        "region": region,
        "churn": churn
    })

    df.to_csv(save_path, index=False)
    print(f"Dataset created successfully: {save_path}")
    return df