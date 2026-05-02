"""Utility functions for the Credit Risk Engine project."""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import pickle

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data preprocessing and transformation."""

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Handle missing values in dataset."""
        df_copy = df.copy()
        if strategy == "mean":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                df_copy[numeric_cols].mean()
            )
        elif strategy == "median":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                df_copy[numeric_cols].median()
            )
        elif strategy == "mode":
            df_copy = df_copy.fillna(df_copy.mode().iloc[0])

        logger.info(f"Missing values handled using {strategy} strategy")
        return df_copy

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame, columns: List[str], method: str = "iqr"
    ) -> pd.DataFrame:
        """Detect and handle outliers."""
        df_copy = df.copy()

        if method == "iqr":
            for col in columns:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)

        logger.info(f"Outliers detected and handled using {method} method")
        return df_copy


class MetricsCalculator:
    """Calculate model performance metrics."""

    @staticmethod
    def get_classification_metrics(y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba[:, 1]),
        }

        return metrics


def save_model(model, filepath: str):
    """Save model to disk."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(filepath: str):
    """Load model from disk."""
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
