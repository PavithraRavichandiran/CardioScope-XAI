"""
XGBoost classifier for CardioScope-XAI.

Trains on the processed UCI Cleveland feature matrix.
Optimises recall to minimise false negatives in clinical screening.
Logs all experiments to MLflow.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

MODEL_PATH      = Path("models/xgboost_model.pkl")
MLFLOW_TRACKING = "mlflow"
EXPERIMENT_NAME = "CardioScope-XAI / XGBoost"
RANDOM_STATE    = 42

# Best hyperparameters (tuned for recall on Cleveland dataset)
DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.2,   # slight upweight on positive class to improve recall
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_STATE,
}


def build_model(params: dict | None = None) -> XGBClassifier:
    """Return an XGBClassifier with given or default parameters."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return XGBClassifier(**p)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
    log_mlflow: bool = True,
) -> tuple[XGBClassifier, dict]:
    """
    Train XGBoost, evaluate on test set, log to MLflow, save model.

    Returns:
        (fitted model, metrics dict)
    """
    model = build_model(params)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_baseline"):
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "recall":    round(recall_score(y_test, y_pred),     4),
            "precision": round(precision_score(y_test, y_pred),  4),
            "f1":        round(f1_score(y_test, y_pred),         4),
            "auc_roc":   round(roc_auc_score(y_test, y_proba),   4),
        }

        if log_mlflow:
            mlflow.log_params({k: v for k, v in (params or DEFAULT_PARAMS).items()
                               if k not in ("use_label_encoder", "eval_metric")})
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # Save locally
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"[xgboost_model] Model saved → {MODEL_PATH}")

    return model, metrics


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
) -> dict:
    """Run stratified k-fold cross-validation and return mean ± std metrics."""
    model = build_model(params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    summary = {}
    for metric in scoring:
        vals = results[f"test_{metric}"]
        summary[metric] = {"mean": round(vals.mean(), 4), "std": round(vals.std(), 4)}

    return summary


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return full evaluation metrics and print classification report."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "recall":    round(recall_score(y_test, y_pred),     4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "f1":        round(f1_score(y_test, y_pred),         4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba),   4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n" + classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
    return metrics


def load_model(path: str | Path = MODEL_PATH) -> XGBClassifier:
    """Load a saved XGBoost model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def predict_proba_single(model: XGBClassifier, X_row: pd.DataFrame) -> float:
    """Return disease probability for a single patient row."""
    return float(model.predict_proba(X_row)[:, 1][0])
