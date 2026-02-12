from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import joblib
from .database import SessionLocal, Prediction
import logging
logging.basicConfig(level=logging.INFO)

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

    # Convert request to dictionary
    data = invoice.dict()
    
    
    # 🔹 Log incoming request
    logging.info(f"Prediction requested: {data}")

    # Create feature list in correct order
    features = [[
        data["invoice_amount"],
        data["avg_delay_days"],
        data["num_past_invoices"],
        data["invoice_gap_days"],
        data["industry_category"],
        data["reliability_score"]
    ]]

    # Get probability
    probability = model.predict_proba(features)[0][1]
    
    # 🔹 Log model output
    logging.info(f"Probability: {probability}")

    # Decision logic
    if probability > 0.7:
        reminder = "Send early reminder"
        tone = "Firm"
    else:
        reminder = "Normal reminder"
        tone = "Friendly"
     # 🔥 ADD DATABASE CODE HERE (BEFORE RETURN)

    db = SessionLocal()

    new_prediction = Prediction(
        invoice_amount=data["invoice_amount"],
        probability=float(probability),
        tone=tone,
        model_version="v1.0"
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    return {
        "late_payment_probability": float(probability),
        "recommended_action": reminder,
        "tone": tone,
        "model_version": "v1.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
      