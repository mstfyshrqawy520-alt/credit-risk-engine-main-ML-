"""Unit tests for Credit Risk Engine pipeline."""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import DataIngestion
from src.utils import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.evaluate import compute_metrics
from src.fairness_analysis import FairnessAnalyzer

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "credit_risk": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "age": [25, 45, 33, 55, 28, 60, 40, 35, 50, 22],
            "income": [
                50000,
                80000,
                60000,
                120000,
                45000,
                90000,
                70000,
                55000,
                85000,
                40000,
            ],
            "credit_score": [650, 720, 580, 800, 620, 760, 690, 640, 710, 600],
            "employment_years": [2, 10, 5, 20, 1, 15, 7, 3, 12, 0],
            "debt_amount": [
                10000,
                5000,
                30000,
                0,
                25000,
                8000,
                15000,
                20000,
                6000,
                35000,
            ],
            "payment_history": [6, 24, 3, 60, 2, 48, 12, 8, 36, 1],
        }
    )


@pytest.fixture
def csv_path(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


# ─── DataIngestion ────────────────────────────────────────────────────────────


def test_load_data(csv_path):
    ing = DataIngestion(csv_path)
    df = ing.load_data()
    assert df.shape[0] == 10


def test_validate_data_passes(csv_path):
    ing = DataIngestion(csv_path)
    ing.load_data()
    assert ing.validate_data() is True


def test_train_test_split(csv_path):
    ing = DataIngestion(csv_path, test_size=0.3)
    (X_tr, y_tr), (X_te, y_te) = ing.get_train_test_split()
    assert len(X_tr) + len(X_te) == 10


def test_validate_missing_col(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    ing = DataIngestion(str(p))
    ing.load_data()
    assert ing.validate_data() is False


# ─── DataProcessor ────────────────────────────────────────────────────────────


def test_missing_value_mean(sample_df):
    sample_df.loc[0, "income"] = np.nan
    result = DataProcessor.handle_missing_values(sample_df, "mean")
    assert result["income"].isna().sum() == 0


def test_outlier_clipping(sample_df):
    sample_df.loc[0, "income"] = 999_999_999
    result = DataProcessor.detect_outliers(sample_df, ["income"])
    assert result["income"].max() < 999_999_999


# ─── FeatureEngineer ──────────────────────────────────────────────────────────


def test_feature_engineer_shape(sample_df):
    eng = FeatureEngineer()
    X = eng.fit_transform(sample_df.drop("credit_risk", axis=1))
    assert X.shape[0] == 10
    assert X.shape[1] == len(eng.feature_names)


def test_feature_engineer_transform(sample_df):
    eng = FeatureEngineer()
    eng.fit_transform(sample_df.drop("credit_risk", axis=1))
    X_new = eng.transform(sample_df.drop("credit_risk", axis=1))
    assert X_new.shape == (10, len(eng.feature_names))


def test_feature_engineer_save_load(tmp_path, sample_df):
    eng = FeatureEngineer()
    eng.fit_transform(sample_df.drop("credit_risk", axis=1))
    path = str(tmp_path / "eng.joblib")
    eng.save(path)
    eng2 = FeatureEngineer.load(path)
    X = eng2.transform(sample_df.drop("credit_risk", axis=1))
    assert X.shape[0] == 10


# ─── Metrics ─────────────────────────────────────────────────────────────────


def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]]
    )
    m = compute_metrics(y_true, y_pred, y_proba)
    assert "roc_auc" in m
    assert "gini" in m
    assert "ks_statistic" in m
    assert 0 <= m["roc_auc"] <= 1


# ─── Fairness ─────────────────────────────────────────────────────────────────


def test_disparate_impact():
    fa = FairnessAnalyzer()
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    sensitive = np.array([1, 1, 1, 0, 0, 0])
    result = fa.disparate_impact(y_pred, sensitive, privileged=1)
    assert "ratio" in result
    assert "compliant" in result


def test_fairness_full_report():
    np.random.seed(0)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    ages = np.random.randint(18, 80, n)
    report = FairnessAnalyzer().full_report(y_true, y_pred, ages)
    assert "all_compliant" in report
    assert "age_disparate_impact" in report
