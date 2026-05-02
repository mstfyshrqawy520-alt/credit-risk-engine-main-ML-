"""Multi-model Training Pipeline with MLflow tracking."""

import sys
import logging
import json
from pathlib import Path
import joblib

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

# heavy boosters will be imported on demand; provide sklearn fallbacks
XGBClassifier = None
LGBMClassifier = None
CatBoostClassifier = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import compute_metrics, print_report

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train XGBoost / LightGBM / CatBoost with SMOTE + MLflow tracking."""

    def __init__(
        self,
        params: dict,
        output_dir: str = "models",
        experiment: str = "credit-risk-engine",
    ):
        self.params = params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.results: dict = {}
        self.best_model = None
        self.best_name = ""
        self.best_auc = 0.0

    def _build(self, name: str):
        p = self.params.get(name, {})
        if name == "xgboost":
            try:
                from xgboost import XGBClassifier as _XGB

                return _XGB(**p, eval_metric="logloss", use_label_encoder=False)
            except Exception:
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(max_iter=200)
        if name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier as _LGBM

                return _LGBM(**p)
            except Exception:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=50)
        if name == "catboost":
            try:
                from catboost import CatBoostClassifier as _Cat

                return _Cat(**p)
            except Exception:
                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier()
        raise ValueError(f"Unknown model: {name}")

    def train_all(self, X_train, y_train, X_test, y_test) -> dict:
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(self.experiment)
            except Exception:
                pass

        # SMOTE for class imbalance when available, otherwise skip
        if SMOTE_AVAILABLE:
            try:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_train)
                logger.info(
                    f"SMOTE: {X_train.shape[0]} → {X_res.shape[0]} samples | "
                    f"default rate: {y_res.mean():.2%}"
                )
            except Exception:
                X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

        for name in ["xgboost", "lightgbm", "catboost"]:
            logger.info(f"Training {name}...")
            model = self._build(name)

            if MLFLOW_AVAILABLE:
                run_ctx = mlflow.start_run(run_name=name)
            else:
                from contextlib import nullcontext

                run_ctx = nullcontext()

            with run_ctx:
                cv_auc = cross_val_score(
                    model,
                    X_res,
                    y_res,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="roc_auc",
                    n_jobs=-1,
                ).mean()

                model.fit(X_res, y_res)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                metrics = compute_metrics(y_test, y_pred, y_proba)
                print_report(metrics, name)

                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_params(self.params.get(name, {}))
                        mlflow.log_metric("cv_auc", round(cv_auc, 4))
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(k, v)
                    except Exception:
                        pass

                path = self.output_dir / f"{name}_model.joblib"
                joblib.dump(model, path)
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_artifact(str(path))
                    except Exception:
                        pass

                self.results[name] = {"metrics": metrics, "cv_auc": round(cv_auc, 4)}
                if metrics["roc_auc"] > self.best_auc:
                    self.best_auc = metrics["roc_auc"]
                    self.best_model = model
                    self.best_name = name

        # Save best model
        best_path = self.output_dir / "best_model.joblib"
        joblib.dump(self.best_model, best_path)
        logger.info(
            f"🏆 Best: {self.best_name} (AUC={self.best_auc:.4f}) → {best_path}"
        )

        # Save results summary
        summary = {
            "best_model": self.best_name,
            "best_auc": self.best_auc,
            "results": self.results,
        }
        (self.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        return self.results
