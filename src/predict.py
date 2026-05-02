"""Inference Engine — loads model and runs predictions."""

import sys
import logging
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

RISK_THRESHOLDS = {"low": 0.35, "moderate": 0.70}


def _categorize(score: float) -> tuple[str, str]:
    if score < RISK_THRESHOLDS["low"]:
        return "LOW RISK", "APPROVE"
    if score < RISK_THRESHOLDS["moderate"]:
        return "MODERATE RISK", "APPROVE"
    return "HIGH RISK", "REVIEW"


class Predictor:
    """Load model + feature engineer and serve predictions."""

    def __init__(self, model_path: str, engineer_path: str):
        self.model = joblib.load(model_path)
        self.engineer: FeatureEngineer = FeatureEngineer.load(engineer_path)
        logger.info(f"Predictor loaded: {model_path}")

    def predict(self, data: dict) -> dict:
        import time

        t0 = time.time()
        df = pd.DataFrame([data])
        X = self.engineer.transform(df)
        proba = self.model.predict_proba(X)[0]
        risk_score = float(proba[1])
        category, decision = _categorize(risk_score)
        return {
            "risk_score": round(risk_score, 4),
            "risk_probability": {
                "low": round(float(proba[0]), 4),
                "high": round(float(proba[1]), 4),
            },
            "risk_category": category,
            "decision": decision,
            "confidence": round(float(max(proba)), 4),
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }
