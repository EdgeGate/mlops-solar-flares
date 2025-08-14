import os
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator

# Swagger UI local (pas de CDN)
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
# swagger-ui-bundle a changé le nom de l’export selon les versions
try:
    from swagger_ui_bundle import swagger_ui_3_path as swagger_ui_path
except ImportError:  # >=1.x
    from swagger_ui_bundle import swagger_ui_path

from model_store import ModelStore

# ======================
# Config
# ======================
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "solar-flares-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
AUTH_TOKEN = os.getenv("API_TOKEN")  # optionnel

mlflow.set_tracking_uri(MLFLOW_URI)

# Charge le modèle via alias (nouveau registry MLflow) + auto-refresh
store = ModelStore(MODEL_NAME, alias=MODEL_ALIAS, refresh_secs=60)

# Crée l'app
app = FastAPI(title="Solar Flares API", version="1.4")

# Swagger-UI servi localement (évite le CDN qui peut être bloqué)
app.mount("/static", StaticFiles(directory=swagger_ui_path), name="static")

# Instrumentation Prometheus : ajouter le middleware AVANT le startup
instrumentator = Instrumentator()
instrumentator.instrument(app)

# ======================
# Sécurité simple (optionnelle)
# ======================
def check_auth(request: Request):
    if not AUTH_TOKEN:
        return
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ======================
# Schémas d'API
# ======================
Instance = Union[Dict[str, Any], List[Any]]

class PredictRequest(BaseModel):
    # Chaque instance peut être:
    # - un dict (features nommées) ou
    # - une list (features positionnelles)
    instances: List[Instance] = Field(..., description="Lignes de features")

class PredictResponse(BaseModel):
    model_version: int
    predictions: List[Any]

# ======================
# Helpers
# ======================
def _normalize_instances_to_dataframe(instances: List[Instance]) -> pd.DataFrame:
    """
    Convertit un mélange de dicts et de listes en DataFrame.
    - Si au moins une instance est une list -> on convertit tout en lignes positionnelles.
      Pour les dicts, on tente d'ordonner sur des clés numériques '0'..'n' ou 0..n.
    - Sinon (tous dicts) -> DataFrame classique par colonnes.
    """
    if not instances:
        raise ValueError("instances is empty")

    has_list = any(isinstance(x, list) for x in instances)
    has_dict = any(isinstance(x, dict) for x in instances)

    if has_list:
        rows: List[List[Any]] = []
        for row in instances:
            if isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict):
                try:
                    as_int_keys = {int(k): v for k, v in row.items()}
                    ordered = [as_int_keys[i] for i in sorted(as_int_keys.keys())]
                    rows.append(ordered)
                except Exception:
                    rows.append(list(row.values()))
            else:
                raise ValueError("Each instance must be a list or a dict")
        return pd.DataFrame(rows)

    if has_dict and not has_list:
        return pd.DataFrame(instances)

    raise ValueError("Unsupported instance types")

# ======================
# Lifecycle
# ======================
@app.on_event("startup")
def _startup():
    # Expose /metrics au démarrage (middleware déjà ajouté au chargement)
    instrumentator.expose(app, endpoint="/metrics")
    # Démarre le rafraîchissement du modèle
    store.start_auto_refresh()

# ======================
# Endpoints
# ======================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    # si le thread n'a pas encore chargé, on tente un chargement synchrone
    v = store.version
    if v < 0:
        try:
            store._load_current()  # force le premier chargement
            v = store.version
        except Exception as e:
            raise HTTPException(503, f"Model not ready: {e}")
    return {"ready": True, "model_version": v}

@app.get("/model-info")
def model_info():
    return {
        "tracking_uri": MLFLOW_URI,
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_version": store.version,
    }

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_auth)])
def predict(body: PredictRequest):
    try:
        df = _normalize_instances_to_dataframe(body.instances)

        # Cast automatique en float64 pour respecter la signature [double]
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.number):
                df[c] = df[c].astype("float64")

        yhat = store.predict(df)
        if hasattr(yhat, "tolist"):
            yhat = yhat.tolist()
        return PredictResponse(model_version=store.version, predictions=yhat)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# ======================
# Swagger-UI local (remplace l'UI par défaut)
# ======================
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()
