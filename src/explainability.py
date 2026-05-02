"""SHAP Explainability Module."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Explainer:
    """SHAP TreeExplainer wrapper for credit risk models."""

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            try:
                import shap

                self._explainer = shap.TreeExplainer(self.model)
                self._shap_available = True
            except Exception as e:
                logger.warning(
                    f"SHAP not available: {e}. Using fallback feature importance."
                )
                self._shap_available = False
        return self._explainer

    def shap_values_single(self, X: np.ndarray) -> dict:
        """Return SHAP values for a single sample as {feature: value}."""
        self._get_explainer()

        if self._shap_available:
            explainer = self._explainer
            sv = explainer.shap_values(X)
            # For binary classifiers shap_values returns list[2] or array
            if isinstance(sv, list):
                values = sv[1][0]  # positive class
            else:
                values = sv[0]
            result = {
                "shap_values": dict(
                    zip(self.feature_names, [round(float(v), 6) for v in values])
                ),
                "base_value": round(
                    float(
                        explainer.expected_value
                        if not hasattr(explainer.expected_value, "__len__")
                        else explainer.expected_value[1]
                    ),
                    6,
                ),
            }
            logger.info("SHAP values computed for single sample")
        else:
            # Fallback: Use model's feature importances if available
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                # Normalize to [-1, 1] range for interpretability
                norm_importances = (importances / importances.sum()) * 2 - 1
                result = {
                    "shap_values": dict(
                        zip(
                            self.feature_names,
                            [round(float(v), 6) for v in norm_importances],
                        )
                    ),
                    "base_value": 0.5,  # Default midpoint
                }
            else:
                # Minimal fallback with uniform distribution
                uniform_val = 1.0 / len(self.feature_names)
                result = {
                    "shap_values": dict(
                        zip(
                            self.feature_names,
                            [round(uniform_val, 6)] * len(self.feature_names),
                        )
                    ),
                    "base_value": 0.5,
                }
            logger.info("Using fallback feature importance (SHAP unavailable)")
        return result

    def top_features(self, X: np.ndarray, top_n: int = 5) -> list:
        """Return top N features by |SHAP value|."""
        sv_dict = self.shap_values_single(X)["shap_values"]
        sorted_feats = sorted(sv_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_feats[:top_n]
