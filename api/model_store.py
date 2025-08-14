# model_store.py
import os
import time
import threading
import pandas as pd
import mlflow
from mlflow import MlflowClient

class ModelStore:
    def __init__(self, model_name: str, alias: str = "Production", refresh_secs: int = 60):
        self.model_name = model_name
        self.alias = alias
        self.refresh_secs = refresh_secs
        self.client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        self._model = None
        self._version = None
        self._lock = threading.RLock()

    def _load_current(self):
        # Récupère la version pointée par l'alias (nouveau registry MLflow)
        mv = self.client.get_model_version_by_alias(self.model_name, self.alias)
        uri = f"models:/{self.model_name}@{self.alias}"
        model = mlflow.pyfunc.load_model(uri)
        with self._lock:
            self._model = model
            self._version = int(mv.version)
        print(f"[ModelStore] Loaded {self.model_name}@{self.alias} (v{mv.version})")

    def predict(self, df: pd.DataFrame):
        with self._lock:
            if self._model is None:
                raise RuntimeError("Modèle non chargé")
            return self._model.predict(df)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version or -1

    def start_auto_refresh(self):
        # premier chargement; si l'alias n'existe pas encore, on attend la boucle
        try:
            self._load_current()
        except Exception as e:
            print(f"[model-refresh] Waiting for alias '{self.alias}' to be set: {e}")

        def loop():
            while True:
                try:
                    mv = self.client.get_model_version_by_alias(self.model_name, self.alias)
                    if self._version != int(mv.version):
                        self._load_current()
                except Exception as e:
                    print(f"[model-refresh] Error: {e}")
                time.sleep(self.refresh_secs)

        threading.Thread(target=loop, daemon=True).start()
