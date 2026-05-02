"""Data Ingestion Module for Credit Risk Engine."""

import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data loading and initial validation."""

    def __init__(self, data_path: str, test_size: float = 0.2):
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.data = None
        self.target_col = "credit_risk"

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(
                f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns"
            )
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self) -> bool:
        """Validate data quality."""
        if self.data is None:
            self.load_data()

        # Check for required columns
        required_cols = ["credit_risk", "age", "income", "credit_score"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return False

        logger.info("Data validation passed")
        return True

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if self.data is None:
            self.load_data()

        from sklearn.model_selection import train_test_split

        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        return (X_train, y_train), (X_test, y_test)
