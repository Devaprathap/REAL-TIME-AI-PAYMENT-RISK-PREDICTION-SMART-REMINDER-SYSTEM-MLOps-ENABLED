from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/xgboost_model.pkl")


class Invoice(BaseModel):
    invoice_amount: float
    avg_delay_days: float
    num_past_invoices: float
    invoice_gap_days: float
    industry_category: float
    reliability_score: float

@app.post("/predict")
def predict(invoice: Invoice):
data = invoice.dict()

df = pd.DataFrame([[
    data["invoice_amount"],
    data["avg_delay_days"],
    data["num_past_invoices"],
    data["invoice_gap_days"],
    data["industry_category"],
    data["reliability_score"]
]], columns=[
    "invoice_amount",
    "avg_delay_days",
    "num_past_invoices",
    "invoice_gap_days",
    "industry_category",
    "reliability_score"
])
