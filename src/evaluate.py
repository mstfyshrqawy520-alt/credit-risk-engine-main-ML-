"""Model Evaluation Metrics Module."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute comprehensive credit-risk metrics."""
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    roc_auc = roc_auc_score(y_true, proba_pos)
    ks_stat, _ = ks_2samp(proba_pos[y_true == 1], proba_pos[y_true == 0])

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4),
        "gini": round(float(2 * roc_auc - 1), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "brier_score": round(float(brier_score_loss(y_true, proba_pos)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    logger.info(
        f"AUC={metrics['roc_auc']} | Gini={metrics['gini']} | KS={metrics['ks_statistic']}"
    )
    return metrics


def print_report(metrics: dict, name: str = "Model"):
    sep = "=" * 48
    print(f"\n{sep}\n  {name}\n{sep}")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<18}: {v}")
    print(sep + "\n")
