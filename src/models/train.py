import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


# ==============================
# 1️⃣ Load Data
# ==============================

df = pd.read_csv("data/invoices.csv")

X = df.drop("late_payment", axis=1)
y = df["late_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

mlflow.set_experiment("CollectIQ_Payment_Risk")


# ==============================
# 2️⃣ Train Model V1 (Baseline)
# ==============================

with mlflow.start_run(run_name="XGBoost_v1"):

    model_v1 = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model_v1.fit(X_train, y_train)

    y_pred_v1 = model_v1.predict_proba(X_test)[:, 1]

    auc_v1 = roc_auc_score(y_test, y_pred_v1)

    mlflow.log_param("version", "v1")
    mlflow.log_metric("roc_auc", auc_v1)
    mlflow.xgboost.log_model(model_v1, name="model_v1")

    joblib.dump(model_v1, "models/xgboost_model.pkl")

    print("✅ V1 ROC-AUC:", auc_v1)


# ==============================
# 3️⃣ Train Model V2 (Improved)
# ==============================

with mlflow.start_run(run_name="XGBoost_v2"):

    model_v2 = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model_v2.fit(X_train, y_train)

    y_pred_v2 = model_v2.predict_proba(X_test)[:, 1]

    auc_v2 = roc_auc_score(y_test, y_pred_v2)

    mlflow.log_param("version", "v2")
    mlflow.log_metric("roc_auc", auc_v2)
    mlflow.xgboost.log_model(model_v2, name="model_v2")

    joblib.dump(model_v2, "models/xgboost_model_v2.pkl")

    print("✅ V2 ROC-AUC:", auc_v2)


# ==============================
# 4️⃣ Compare
# ==============================

print("\n📊 MODEL COMPARISON")
print("V1 ROC-AUC:", auc_v1)
print("V2 ROC-AUC:", auc_v2)

if auc_v2 > auc_v1:
    print("🚀 V2 is better!")
else:
    print("📌 V1 performs better.")
    
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("CollectIQ_Payment_Risk")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"]
)

best_run = runs[0]
best_model_name = best_run.data.params.get("version")

print("🏆 Best Model:", best_model_name)
