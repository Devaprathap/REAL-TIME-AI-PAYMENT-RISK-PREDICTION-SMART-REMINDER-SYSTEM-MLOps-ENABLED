1. Project Title

Real-Time AI-Based Payment Risk Prediction & Smart Reminder System with MLOps

2. Problem Statement

Service businesses face delayed payments which affect cash flow.
Traditional systems send fixed reminders without predicting risk.

This system:

Predicts late payment probability

Recommends reminder strategy

Logs predictions

Monitors live system analytics

Supports model versioning

3. Architecture Overview


    1.Invoice data → FastAPI

    2.XGBoost model predicts probability

    3.Response includes:

        Risk %
        Action
        Tone

    4.Prediction stored in database

    5.Streamlit dashboard displays:

        Risk gauge
        Live analytics

    6.Deployed on Render

4. Tech Stack

        Python

        XGBoost

        FastAPI

        SQLite

        Streamlit

        Joblib

        Render (Deployment)

 5. API Endpoints

        POST /predict
        GET /stats
        GET /health
        GET /feature-importance


 6. Deployment Link

 https://dashboard.render.com/web/srv-d66nr8v5r7bs739bm6ug
   
   https://collectiq-api.onrender.com
    