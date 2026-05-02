"""Feature Engineering Module for Credit Risk Engine."""

import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age",
    "income",
    "credit_score",
    "employment_years",
    "debt_amount",
    "payment_history",
    "debt_to_income",
    "income_stability",
    "payment_risk_score",
    "credit_utilization",
    "age_income_interaction",
]


class FeatureEngineer:
    """Create and scale features for credit risk modeling."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.feature_names = FEATURE_NAMES
        self._fitted = False

    @staticmethod
    def _derive(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["debt_to_income"] = df["debt_amount"] / (df["income"] + 1)
        df["income_stability"] = df["income"] / (df["employment_years"] + 1)
        df["payment_risk_score"] = 1.0 / (df["payment_history"] + 1)
        df["credit_utilization"] = (850 - df["credit_score"]) / 550.0
        df["age_income_interaction"] = df["age"] * df["income"] / 1e6
        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = self._derive(df)
        X = df_feat[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        logger.info(f"FeatureEngineer fitted: {X_scaled.shape}")
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit_transform first.")
        df_feat = self._derive(df)
        return self.scaler.transform(df_feat[self.feature_names].values)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        return joblib.load(path)
