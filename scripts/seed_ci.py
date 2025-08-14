# scripts/seed_ci.py
import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "solar-flares-classifier"

mlflow.set_tracking_uri(TRACKING)
mlflow.set_experiment("ci-seed")

# Train un mini modèle
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)

# Récupère la dernière version et assigne l'alias Production
client = MlflowClient()
latest = client.get_latest_versions(MODEL_NAME)
version = latest[-1].version if latest else "1"
client.set_registered_model_alias(MODEL_NAME, "Production", version)
print(f"Seeded {MODEL_NAME} v{version} -> alias Production")
