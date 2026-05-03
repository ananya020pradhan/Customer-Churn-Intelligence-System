from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_customer

app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    usage_hours: float
    login_count: int
    support_tickets: int
    payment_delay_days: int
    contract_type: str
    internet_service: str
    is_autopay: str
    region: str


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerInput):
    customer_data = data.dict()
    result = predict_customer(customer_data)
    return result