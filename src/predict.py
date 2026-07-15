"""
Unified inference pipeline for CardioScope-XAI.

Single entry point that:
  1. Takes raw patient clinical values (dict) + optional temporal sequence
  2. Runs feature engineering → XGBoost features → LSTM embedding → fusion
  3. Returns risk probability, risk tier, SHAP explanation

Used by the Streamlit dashboard for all predictions.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.feature_engineering import build_feature_matrix, engineer_features, encode_categoricals
from src.xgboost_model import load_model as load_xgb, predict_proba_single
from src.lstm_model import load_encoder, extract_embeddings
from src.fusion import load_model as load_fusion, predict_single, RISK_LABELS, RISK_COLORS
from src.explainability import build_explainer, get_patient_explanation
from src.temporal_data import generate_temporal_sequences, normalize_sequences, N_TIMESTEPS

SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.pkl"

# ── Model cache (loaded once per session) ─────────────────────────────────────

_cache: dict = {}

def _load_models() -> dict:
    """Load all models once and cache them."""
    if not _cache:
        _cache["xgb"]     = load_xgb()
        _cache["encoder"] = load_encoder()
        _cache["fusion"]  = load_fusion()
        _cache["scaler"]  = joblib.load(SCALER_PATH)
        _cache["explainer"] = build_explainer(_cache["xgb"])
    return _cache


# ── Input helpers ─────────────────────────────────────────────────────────────

CLINICAL_DEFAULTS = {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 130,
    "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
}

def make_patient_df(clinical_inputs: dict) -> pd.DataFrame:
    """Convert a dict of raw clinical values into a single-row DataFrame."""
    row = {**CLINICAL_DEFAULTS, **clinical_inputs}
    return pd.DataFrame([row])


# ── Core prediction function ──────────────────────────────────────────────────

def predict(
    clinical_inputs: dict,
    temporal_sequence: np.ndarray | None = None,
) -> dict:
    """
    Full inference pipeline for one patient.

    Args:
        clinical_inputs   : dict of raw clinical values (13 UCI features)
        temporal_sequence : (12, 4) normalised sequence — if None, auto-generated

    Returns:
        dict with:
            probability  : float — disease probability from fusion model
            risk_tier    : int   — 0=Low, 1=Medium, 2=High
            risk_label   : str
            risk_color   : str (hex)
            xgb_prob     : float — XGBoost-only probability
            shap_df      : pd.DataFrame — top 5 SHAP drivers
            clinical_df  : pd.DataFrame — processed clinical feature row
    """
    models = _load_models()

    # 1. Build clinical feature row
    raw_df      = make_patient_df(clinical_inputs)
    raw_df["target"] = 0  # placeholder, dropped by build_feature_matrix

    X_clinical, _ = build_feature_matrix(raw_df, fit_scaler=False, scaler=models["scaler"])

    # 2. XGBoost-only probability
    xgb_prob = predict_proba_single(models["xgb"], X_clinical)

    # 3. LSTM temporal embedding
    if temporal_sequence is None:
        temporal_sequence = _generate_sequence_for_patient(raw_df, clinical_inputs)

    embedding = models["encoder"].predict(
        temporal_sequence[np.newaxis, ...], verbose=0
    )  # (1, 16)

    # 4. Fusion risk score
    result = predict_single(models["fusion"], X_clinical.values, embedding)

    # 5. SHAP explanation
    shap_vals = models["explainer"].shap_values(X_clinical)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    base_val = (models["explainer"].expected_value
                if not isinstance(models["explainer"].expected_value, list)
                else models["explainer"].expected_value[1])

    shap_df = get_patient_explanation(shap_vals, float(base_val), X_clinical,
                                      patient_index=0, top_n=5)

    return {
        **result,
        "xgb_prob":   round(xgb_prob, 4),
        "shap_df":    shap_df,
        "clinical_df": X_clinical,
    }


def _generate_sequence_for_patient(raw_df: pd.DataFrame, inputs: dict) -> np.ndarray:
    """Auto-generate a 12-step temporal sequence from static UCI values."""
    # Attach a dummy target to drive trend direction (use XGBoost probability as proxy)
    models    = _load_models()
    X_temp, _ = build_feature_matrix(raw_df, fit_scaler=False, scaler=models["scaler"])
    prob      = predict_proba_single(models["xgb"], X_temp)
    raw_df    = raw_df.copy()
    raw_df["target"] = int(prob > 0.5)

    seqs, _ = generate_temporal_sequences(raw_df, n_timesteps=N_TIMESTEPS)
    seqs_norm, _ = normalize_sequences(seqs)
    return seqs_norm[0]  # (12, 4)


# ── What-if simulation ────────────────────────────────────────────────────────

def simulate_what_if(base_inputs: dict, modifications: dict) -> dict:
    """
    Re-run inference with modified clinical values.

    Args:
        base_inputs   : original patient dict
        modifications : dict of overrides e.g. {"trestbps": 120, "chol": 200}

    Returns:
        Same structure as predict(), plus "delta_probability"
    """
    base_result     = predict(base_inputs)
    modified_inputs = {**base_inputs, **modifications}
    new_result      = predict(modified_inputs)

    delta = round(new_result["probability"] - base_result["probability"], 4)
    new_result["delta_probability"] = delta
    new_result["base_probability"]  = base_result["probability"]

    return new_result
