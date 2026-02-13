from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import logging
from .database import SessionLocal, Prediction

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CollectIQ API")

# ==============================
# LOAD MODELS
# ==============================

model_v1 = joblib.load("models/xgboost_model.pkl")
model_v2 = joblib.load("models/xgboost_model_v2.pkl")

# Default best model (can later auto-pick via MLflow)
BEST_MODEL = "v2"


# ==============================
# REQUEST SCHEMA
# ==============================

class Invoice(BaseModel):
    invoice_amount: float
    avg_delay_days: float
    num_past_invoices: float
    invoice_gap_days: float
    industry_category: float
    reliability_score: float


# ==============================
# PREDICT ENDPOINT
# ==============================

@app.post("/predict")
def predict(invoice: Invoice, model_version: str = Query(None)):

    selected_model = model_version if model_version else BEST_MODEL

    data = invoice.dict()

    features = [[
        data["invoice_amount"],
        data["avg_delay_days"],
        data["num_past_invoices"],
        data["invoice_gap_days"],
        data["industry_category"],
        data["reliability_score"]
    ]]

    if selected_model == "v1":
        probability = model_v1.predict_proba(features)[0][1]
    else:
        probability = model_v2.predict_proba(features)[0][1]
        selected_model = "v2"

    # Risk Logic
    recommended_action = "Early reminder" if probability > 0.7 else "Normal reminder"
    tone = "Firm" if probability > 0.7 else "Friendly"

    # ==============================
    # SAVE TO DATABASE
    # ==============================

    db = SessionLocal()

    new_prediction = Prediction(
        invoice_amount=data["invoice_amount"],
        probability=float(probability),
        tone=tone,
        model_version=selected_model
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    logging.info(f"Prediction: {probability} | Model: {selected_model}")

    return {
        "late_payment_probability": float(probability),
        "recommended_action": recommended_action,
        "tone": tone,
        "model_version": selected_model
    }


# ==============================
# STATS ENDPOINT
# ==============================

@app.get("/stats")
def get_stats():

    db = SessionLocal()
    predictions = db.query(Prediction).all()
    total = len(predictions)

    if total == 0:
        db.close()
        return {
            "total_predictions": 0,
            "average_risk": 0,
            "high_risk_predictions": 0
        }

    avg_probability = sum(p.probability for p in predictions) / total
    high_risk = len([p for p in predictions if p.probability > 0.7])

    db.close()

    return {
        "total_predictions": total,
        "average_risk": round(avg_probability, 3),
        "high_risk_predictions": high_risk
    }


# ==============================
# FEATURE IMPORTANCE
# ==============================

@app.get("/feature-importance")
def feature_importance():

    try:
        booster = model_v2.get_booster()
        importance = booster.get_score(importance_type="weight")

        return {"feature_importance": importance}

    except Exception:
        return {"error": "Feature importance unavailable"}


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "best_model": BEST_MODEL,
        "models_loaded": True
    }
