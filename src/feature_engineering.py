"""
Clinical feature engineering for CardioScope-XAI.

Derives clinically meaningful cardiovascular indicators from UCI raw features
and prepares the final feature matrix for XGBoost training.

Engineered features:
    - map_value       : Mean Arterial Pressure  (diastolic proxy from trestbps)
    - pulse_pressure  : Systolic − Diastolic proxy (trestbps range indicator)
    - chol_hr_ratio   : Cholesterol / Max Heart Rate (lipid-to-cardiac stress)
    - hr_reserve      : Max HR achieved vs age-predicted max (220 − age)
    - hr_reserve_pct  : hr_reserve as percentage of age-predicted max
    - st_risk_index   : oldpeak × exang (ST depression amplified by angina)
    - age_bp_index    : age × trestbps (combined age-pressure cardiovascular load)

Encoding:
    - One-hot encode: cp, restecg, slope, thal
    - Binary features left as-is: sex, fbs, exang, ca
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.pkl"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically derived cardiovascular features to the dataframe."""
    df = df.copy()

    # Mean Arterial Pressure proxy
    # MAP ≈ DBP + (1/3)(SBP − DBP); using trestbps as SBP, estimating DBP ≈ SBP * 0.65
    df["map_value"] = df["trestbps"] * 0.65 + (df["trestbps"] * 0.35) / 3

    # Pulse pressure proxy (reflects arterial stiffness)
    df["pulse_pressure"] = df["trestbps"] * 0.35

    # Cholesterol to max heart rate ratio (higher = worse lipid-cardiac profile)
    df["chol_hr_ratio"] = df["chol"] / (df["thalach"] + 1e-6)

    # Heart rate reserve: how much HR was achieved vs age-predicted max
    age_predicted_max_hr = 220 - df["age"]
    df["hr_reserve"] = df["thalach"] - age_predicted_max_hr
    df["hr_reserve_pct"] = df["thalach"] / (age_predicted_max_hr + 1e-6)

    # ST risk index: ST depression × exercise-induced angina
    df["st_risk_index"] = df["oldpeak"] * df["exang"]

    # Combined age-blood pressure cardiovascular load
    df["age_bp_index"] = df["age"] * df["trestbps"]

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features. Drop first to avoid multicollinearity."""
    df = df.copy()
    categorical_cols = ["cp", "restecg", "slope", "thal"]

    # Ensure integer type before encoding
    for col in categorical_cols:
        df[col] = df[col].astype(int)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df


def scale_continuous(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale continuous + engineered numeric columns.

    Args:
        df    : Feature dataframe (no target column).
        fit   : If True, fit a new scaler and save it. If False, use provided scaler.
        scaler: Pre-fitted scaler (required when fit=False).

    Returns:
        Scaled dataframe and the scaler used.
    """
    df = df.copy()

    # All float columns (engineered + original continuous) get scaled
    float_cols = df.select_dtypes(include=[np.floating, float]).columns.tolist()

    if fit:
        scaler = StandardScaler()
        df[float_cols] = scaler.fit_transform(df[float_cols])
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[feature_engineering] Scaler fitted and saved to {SCALER_PATH}")
    else:
        if scaler is None:
            raise ValueError("Provide a fitted scaler when fit=False.")
        # Use the exact columns the scaler was fitted on
        float_cols = list(scaler.feature_names_in_)
        # Cast to float so transform works regardless of input dtype
        df[float_cols] = df[float_cols].astype(float)
        df[float_cols] = scaler.transform(df[float_cols])
        return df, scaler

    return df, scaler


def load_scaler() -> StandardScaler:
    """Load the saved scaler from disk."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run the preprocessing pipeline first."
        )
    return joblib.load(SCALER_PATH)


FEATURE_COLS_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_columns.pkl"


def build_feature_matrix(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full pipeline: engineer → encode → scale.

    Args:
        df         : Clean UCI dataframe (with target column).
        fit_scaler : Fit a new scaler (True for training, False for inference).
        scaler     : Pre-fitted scaler (used when fit_scaler=False).

    Returns:
        (X_processed, scaler) — X has no target column.
    """
    X = df.drop(columns=["target"], errors="ignore")
    X = engineer_features(X)
    X = encode_categoricals(X)
    X, scaler = scale_continuous(X, fit=fit_scaler, scaler=scaler)

    if fit_scaler:
        # Save column order so inference can reindex to match
        FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(X.columns.tolist(), FEATURE_COLS_PATH)
    else:
        # Reindex to match training columns — fills any missing OHE cols with 0
        if FEATURE_COLS_PATH.exists():
            expected_cols = joblib.load(FEATURE_COLS_PATH)
            X = X.reindex(columns=expected_cols, fill_value=0)

    print(f"[feature_engineering] Final feature matrix: {X.shape[1]} features, {X.shape[0]} samples.")
    return X, scaler
