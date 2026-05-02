"""FastAPI application — Credit Risk Engine REST API."""

import sys
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.security.api_key import APIKeyHeader

# --- path setup ---
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import Predictor
from src.explainability import Explainer
from src.feature_engineering import FEATURE_NAMES

# --- Optional metrics (Prometheus) ---
try:
    from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

    METRICS_AVAILABLE = True
    REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
    REQUEST_LATENCY = Summary(
        "api_request_latency_seconds", "Request latency seconds", ["endpoint"]
    )
except Exception:
    METRICS_AVAILABLE = False

# --- Optional API Key security ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)):
    required = os.getenv("API_KEY")
    if not required:
        return True
    if not api_key or api_key != required:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


# --- logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- app ---
app = FastAPI(
    title="Credit Risk Engine",
    description="AI-powered credit risk scoring with explainability",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- model loading ---
MODEL_PATH = ROOT / "models" / "best_model.joblib"
ENG_PATH = ROOT / "models" / "feature_engineer.joblib"
SUMMARY_PATH = ROOT / "models" / "training_summary.json"

predictor: Optional[Predictor] = None
explainer_obj: Optional[Explainer] = None


@app.on_event("startup")
def load_model():
    global predictor, explainer_obj
    if MODEL_PATH.exists() and ENG_PATH.exists():
        predictor = Predictor(str(MODEL_PATH), str(ENG_PATH))
        explainer_obj = Explainer(predictor.model, FEATURE_NAMES)
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️  Model not found — run scripts/train.py first")


# ─── Schemas ─────────────────────────────────────────────────────────────────


class ApplicationInput(BaseModel):
    age: int = Field(..., ge=18, le=80, example=35)
    income: float = Field(..., gt=0, example=75000)
    credit_score: int = Field(..., ge=300, le=850, example=720)
    employment_years: float = Field(..., ge=0, example=5)
    debt_amount: float = Field(..., ge=0, example=25000)
    payment_history: int = Field(..., ge=0, le=120, example=12)


class PredictionResponse(BaseModel):
    risk_score: float
    risk_probability: dict
    risk_category: str
    decision: str
    confidence: float
    processing_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0",
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    if not SUMMARY_PATH.exists():
        raise HTTPException(404, "Training summary not found")
    return json.loads(SUMMARY_PATH.read_text())


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(data: ApplicationInput, api_key: str = Depends(get_api_key)):
    if predictor is None:
        raise HTTPException(503, "Model not loaded — run scripts/train.py first")
    start = time.time()
    result = predictor.predict(data.model_dump())
    elapsed = time.time() - start
    logger.info(f"Prediction: score={result['risk_score']} | {result['risk_category']}")
    if METRICS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint="predict").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(elapsed)
    return result


@app.post("/explain", tags=["Explainability"])
def explain(data: ApplicationInput):
    if predictor is None or explainer_obj is None:
        raise HTTPException(503, "Model not loaded")
    df = pd.DataFrame([data.model_dump()])
    X = predictor.engineer.transform(df)
    shap_result = explainer_obj.shap_values_single(X)
    top = explainer_obj.top_features(X, top_n=6)
    prediction = predictor.predict(data.model_dump())
    if METRICS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint="explain").inc()
    return {
        "prediction": prediction,
        "shap_values": shap_result["shap_values"],
        "base_value": shap_result["base_value"],
        "top_features": [{"feature": f, "shap_value": round(v, 6)} for f, v in top],
    }


if METRICS_AVAILABLE:

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
