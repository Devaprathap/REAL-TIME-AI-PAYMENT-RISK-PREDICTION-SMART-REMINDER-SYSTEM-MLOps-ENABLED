from fastapi import FastAPI
from fastapi import Query
import joblib
import pandas as pd
from pydantic import BaseModel
import joblib
from .database import SessionLocal, Prediction
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

model = joblib.load("models/xgboost_model.pkl")
model_v1 = joblib.load("models/xgboost_model.pkl")
# later you can add:
#model_v2 = joblib.load("models/lightgbm_model.pkl")
model_v2 = joblib.load("models/xgboost_model_v2.pkl")

class Invoice(BaseModel):
    invoice_amount: float
    avg_delay_days: float
    num_past_invoices: float
    invoice_gap_days: float
    industry_category: float
    reliability_score: float


@app.post("/predict")
def predict(invoice: Invoice, model_version: str = Query("v2")):

    data = invoice.dict()

    features = [[
        data["invoice_amount"],
        data["avg_delay_days"],
        data["num_past_invoices"],
        data["invoice_gap_days"],
        data["industry_category"],
        data["reliability_score"]
    ]]

    # Model selection
    if model_version == "v1":
        probability = model_v1.predict_proba(features)[0][1]
        used_model = "v1"
    else:
        probability = model_v2.predict_proba(features)[0][1]
        used_model = "v2"

    # Business logic
    recommended_action = "Early reminder" if probability > 0.7 else "Normal reminder"
    tone = "Firm" if probability > 0.7 else "Friendly"

    # 🔥 SAVE TO DATABASE
    db = SessionLocal()

    new_prediction = Prediction(
        invoice_amount=data["invoice_amount"],
        probability=float(probability),
        tone=tone,
        model_version=used_model
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    # Return response
    return {
        "late_payment_probability": float(probability),
        "recommended_action": recommended_action,
        "tone": tone,
        "model_version": used_model
    }

        
    
@app.get("/stats")
def get_stats():

    db = SessionLocal()

    predictions = db.query(Prediction).all()

    total_predictions = len(predictions)

    if total_predictions == 0:
        db.close()
        return {"message": "No predictions yet"}

    avg_probability = sum(p.probability for p in predictions) / total_predictions

    high_risk_count = len([p for p in predictions if p.probability > 0.7])

    db.close()

    return {
        "total_predictions": total_predictions,
        "average_risk": round(avg_probability, 3),
        "high_risk_predictions": high_risk_count
    }

@app.get("/feature-importance")
def feature_importance():
    try:
        importances = model.get_booster().get_score(importance_type="weight")

        return {
            "feature_importance": importances
        }
    except:
        return {"error": "Feature importance not available"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
      