import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.xgboost

mlflow.set_experiment("CollectIQ_Payment_Risk")

# Load dataset
df = pd.read_csv("data/invoices.csv")

X = df.drop("late_payment", axis=1)
y = df["late_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different hyperparameter combinations
configs = [
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.2},
]

for config in configs:
    with mlflow.start_run():

        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        print(f"Config: {config} → ROC-AUC: {auc:.4f}")

        # Log parameters
        mlflow.log_params(config)

        # Log metric
        mlflow.log_metric("roc_auc", auc)

        # Log model
        mlflow.xgboost.log_model(model, "model")
