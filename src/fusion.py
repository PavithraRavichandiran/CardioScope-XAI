"""
Feature-level fusion for CardioScope-XAI.

Concatenates:
  - LSTM temporal embeddings  (n_patients, 16)
  - XGBoost clinical features (n_patients, n_clinical_features)

Trains a LogisticRegression head on the fused vector to produce a
unified cardiovascular risk score, mapped to 3 tiers:
  0 = Low Risk    (probability < 0.35)
  1 = Medium Risk (probability 0.35 – 0.65)
  2 = High Risk   (probability > 0.65)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, classification_report
)
import mlflow

FUSION_MODEL_PATH  = Path("models/fusion_model.pkl")
MLFLOW_TRACKING    = "mlflow"
EXPERIMENT_NAME    = "CardioScope-XAI / Fusion"
RANDOM_STATE       = 42

LOW_THRESHOLD    = 0.35
HIGH_THRESHOLD   = 0.65
RISK_LABELS      = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS      = {0: "#4CAF50",  1: "#FF9800",     2: "#F44336"}


def build_fused_matrix(
    clinical_features: pd.DataFrame,
    lstm_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Concatenate XGBoost clinical features with LSTM temporal embeddings.

    Args:
        clinical_features : (n, n_clinical)  — processed feature matrix from build_feature_matrix()
        lstm_embeddings   : (n, 16)          — from lstm_model.extract_embeddings()

    Returns:
        fused : (n, n_clinical + 16)
    """
    if len(clinical_features) != len(lstm_embeddings):
        raise ValueError(
            f"Row mismatch: clinical={len(clinical_features)}, "
            f"embeddings={len(lstm_embeddings)}"
        )
    clinical_arr = clinical_features.values if isinstance(clinical_features, pd.DataFrame) \
                   else clinical_features
    fused = np.concatenate([clinical_arr, lstm_embeddings], axis=1)
    print(f"[fusion] Fused matrix shape: {fused.shape}  "
          f"(clinical={clinical_arr.shape[1]} + temporal={lstm_embeddings.shape[1]})")
    return fused.astype(np.float32)


def train(
    X_fused: np.ndarray,
    y: np.ndarray,
    log_mlflow: bool = True,
) -> tuple[LogisticRegression, dict]:
    """
    Train a LogisticRegression head on the fused feature matrix.

    Returns:
        model   : fitted LogisticRegression
        metrics : dict of evaluation metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",   # boost recall
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="fusion_logistic_regression"):
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "recall":    round(recall_score(y_test, y_pred),     4),
            "precision": round(precision_score(y_test, y_pred),  4),
            "f1":        round(f1_score(y_test, y_pred),         4),
            "auc_roc":   round(roc_auc_score(y_test, y_proba),   4),
        }

        # 5-fold CV on full dataset
        cv_scores = cross_val_score(model, X_fused, y, cv=5, scoring="recall")
        metrics["cv_recall_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_recall_std"]  = round(cv_scores.std(),  4)

        if log_mlflow:
            mlflow.log_params({"model": "LogisticRegression", "C": 0.5,
                               "class_weight": "balanced", "fused_dim": X_fused.shape[1]})
            mlflow.log_metrics(metrics)

        FUSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, FUSION_MODEL_PATH)
        print(f"[fusion] Model saved → {FUSION_MODEL_PATH}")

        print("\n── Fusion Model Metrics ──")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    return model, metrics


def predict_risk(
    model: LogisticRegression,
    X_fused: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict risk probabilities and assign risk tiers.

    Returns:
        probabilities : (n,) float — disease probability
        risk_tiers    : (n,) int  — 0=Low, 1=Medium, 2=High
    """
    probabilities = model.predict_proba(X_fused)[:, 1]
    risk_tiers = np.where(
        probabilities < LOW_THRESHOLD, 0,
        np.where(probabilities > HIGH_THRESHOLD, 2, 1)
    )
    return probabilities, risk_tiers


def predict_single(
    model: LogisticRegression,
    clinical_row: np.ndarray,
    lstm_embedding: np.ndarray,
) -> dict:
    """
    Full fusion inference for one patient.

    Args:
        model          : fitted fusion model
        clinical_row   : (1, n_clinical) — one patient's clinical features
        lstm_embedding : (1, 16)         — one patient's temporal embedding

    Returns:
        dict with probability, risk_tier, risk_label, risk_color
    """
    fused = np.concatenate([clinical_row, lstm_embedding], axis=1)
    prob  = float(model.predict_proba(fused)[:, 1][0])
    tier  = 0 if prob < LOW_THRESHOLD else (2 if prob > HIGH_THRESHOLD else 1)

    return {
        "probability": round(prob, 4),
        "risk_tier":   tier,
        "risk_label":  RISK_LABELS[tier],
        "risk_color":  RISK_COLORS[tier],
    }


def load_model() -> LogisticRegression:
    """Load the saved fusion model."""
    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fusion model not found at {FUSION_MODEL_PATH}. Run training first."
        )
    return joblib.load(FUSION_MODEL_PATH)
