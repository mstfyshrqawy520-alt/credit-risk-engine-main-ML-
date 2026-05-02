"""Fairness Analysis — EU AI Act & ECOA compliant metrics."""

from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


def _tpr(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn + 1e-10)


class FairnessAnalyzer:
    """Compute fairness metrics for sensitive attributes."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def disparate_impact(self, y_pred, sensitive, privileged) -> dict:
        """4/5 rule: unprivileged approval / privileged approval >= 0.80."""
        mask = sensitive == privileged
        rate_priv = y_pred[mask].mean()
        rate_unpriv = y_pred[~mask].mean()
        ratio = float(rate_unpriv) / (float(rate_priv) + 1e-10)
        return {"ratio": round(ratio, 4), "compliant": bool(ratio >= 0.80)}

    def equal_opportunity_diff(self, y_true, y_pred, sensitive, privileged) -> dict:
        """TPR difference between groups should be < 0.05."""
        mask = sensitive == privileged
        diff = abs(
            _tpr(y_true[mask], y_pred[mask]) - _tpr(y_true[~mask], y_pred[~mask])
        )
        return {"difference": round(diff, 4), "compliant": bool(diff < self.threshold)}

    def demographic_parity(self, y_pred, sensitive, privileged) -> dict:
        """Difference in approval rates between groups."""
        mask = sensitive == privileged
        diff = abs(y_pred[mask].mean() - y_pred[~mask].mean())
        return {
            "difference": round(float(diff), 4),
            "compliant": bool(diff < self.threshold),
        }

    def full_report(
        self, y_true, y_pred, ages, default_age_threshold: int = 50
    ) -> dict:
        age_group = (ages > default_age_threshold).astype(int)
        report = {
            "age_disparate_impact": self.disparate_impact(y_pred, age_group, 0),
            "age_equal_opportunity": self.equal_opportunity_diff(
                y_true, y_pred, age_group, 0
            ),
            "age_demographic_parity": self.demographic_parity(y_pred, age_group, 0),
            "overall_default_rate": round(float(y_pred.mean()), 4),
            "threshold_used": self.threshold,
        }
        all_compliant = all(
            v.get("compliant", True) for v in report.values() if isinstance(v, dict)
        )
        report["all_compliant"] = bool(all_compliant)
        logger.info(f"Fairness report generated. All compliant: {all_compliant}")
        return report
