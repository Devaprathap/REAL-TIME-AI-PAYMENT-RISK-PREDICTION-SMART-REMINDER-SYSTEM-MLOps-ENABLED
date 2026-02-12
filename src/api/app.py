from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

model = mlflow.pyfunc.load_model(
    "models:/CollectIQ_Model@production"
)

class Invoice(BaseModel):
    invoice_amount: float
    avg_delay_days: float
    num_past_invoices: float
    invoice_gap_days: float
    industry_category: float
    reliability_score: float

@app.post("/predict")
def predict(data: Invoice):
    df = pd.DataFrame([data.dict()])
    probability = model.predict(df)[0]

    if probability > 0.7:
        reminder_day = 2
        tone = "Firm"
    else:
        reminder_day = 5
        tone = "Friendly"

    return {
        "late_payment_probability": float(probability),
        "recommended_reminder_day": reminder_day,
        "tone": tone
    }
