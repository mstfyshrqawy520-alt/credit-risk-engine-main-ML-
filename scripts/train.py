"""Main Training Script — run from project root."""

import sys
import json
import logging
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Logging ─────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger("train")

# ─── Imports ─────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

from data.generate_data import generate_credit_data
from src.data_ingestion import DataIngestion
from src.utils import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.fairness_analysis import FairnessAnalyzer


def main():
    # ── Config ──────────────────────────────────────────────────────────────
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    params = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text())

    data_path = ROOT / config["data"]["path"]
    model_dir = ROOT / config["model"]["output_dir"]
    model_dir.mkdir(exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    if not data_path.exists():
        logger.info("Generating synthetic dataset...")
        df = generate_credit_data(5000)
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Dataset saved: {data_path}")
    else:
        logger.info(f"Loading existing dataset: {data_path}")

    ingestion = DataIngestion(str(data_path), test_size=config["data"]["test_size"])
    ingestion.load_data()
    ingestion.validate_data()

    df = DataProcessor.handle_missing_values(ingestion.data, strategy="median")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != config["data"]["target_col"]]
    df = DataProcessor.detect_outliers(df, numeric_cols)

    # ── Split ────────────────────────────────────────────────────────────────
    target = config["data"]["target_col"]
    X_raw = df.drop(target, axis=1)
    y = df[target].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    logger.info(
        f"Train: {len(y_train)} | Test: {len(y_test)} | "
        f"Default rate: {y.mean():.2%}"
    )

    # ── Feature Engineering ─────────────────────────────────────────────────
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(X_train_raw)
    X_test = engineer.transform(X_test_raw)
    engineer.save(str(model_dir / "feature_engineer.joblib"))

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = ModelTrainer(
        params=params,
        output_dir=str(model_dir),
        experiment=config["mlflow"]["experiment_name"],
    )
    trainer.train_all(X_train, y_train, X_test, y_test)

    # ── Fairness ─────────────────────────────────────────────────────────────
    y_pred = trainer.best_model.predict(X_test)
    ages = X_test_raw["age"].values
    fairness = FairnessAnalyzer().full_report(y_test, y_pred, ages)
    report_path = model_dir / "fairness_report.json"
    report_path.write_text(json.dumps(fairness, indent=2))
    logger.info(f"Fairness report → {report_path}")
    logger.info(f"All compliant: {fairness['all_compliant']}")

    logger.info("\n✅ Training complete!")
    logger.info(f"   Best model : {trainer.best_name}")
    logger.info(f"   Best AUC   : {trainer.best_auc:.4f}")
    logger.info(f"   Models dir : {model_dir}")


if __name__ == "__main__":
    main()
