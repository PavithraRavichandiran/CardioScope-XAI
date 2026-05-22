"""
UCI Heart Disease dataset loader.

Supports: heart_disease_uci.csv (multi-dataset version with string categoricals)
Filters to Cleveland subset (303 records) to match the project scope.

Columns in source file:
    id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch,
    exang, oldpeak, slope, ca, thal, num
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/heart_disease_uci.csv")

# Final feature columns after cleaning (used by downstream modules)
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
CONTINUOUS_FEATURES  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES      = ["sex", "fbs", "exang", "ca"]

# ── Value maps (string → int) ─────────────────────────────────────────────────

_SEX_MAP     = {"Male": 1, "Female": 0}
_CP_MAP      = {"typical angina": 1, "atypical angina": 2,
                "non-anginal": 3, "asymptomatic": 4}
_RESTECG_MAP = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
_SLOPE_MAP   = {"upsloping": 1, "flat": 2, "downsloping": 3}
_THAL_MAP    = {"normal": 3, "fixed defect": 6, "reversable defect": 7}
_BOOL_MAP    = {"TRUE": 1, "FALSE": 0, True: 1, False: 0}


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the UCI multi-dataset CSV and return the raw DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download heart_disease_uci.csv from:\n"
            "  https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data\n"
            "and place it in data/raw/heart_disease_uci.csv"
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, dataset: str = "Cleveland") -> pd.DataFrame:
    """
    Clean and normalise the UCI multi-dataset file.

    Steps:
        1. Filter to the specified dataset (default: Cleveland)
        2. Drop admin columns (id, dataset)
        3. Map string categoricals → integers
        4. Map TRUE/FALSE → 1/0
        5. Rename thalch → thalach, num → target
        6. Binarise target: 0 = no disease, 1+ = disease → 1
        7. Drop rows with missing values
    """
    df = df.copy()

    # 1. Filter dataset
    if dataset:
        df = df[df["dataset"] == dataset].copy()
        print(f"[data_loader] Using '{dataset}' subset: {len(df)} rows.")

    # 2. Drop admin columns
    df.drop(columns=["id", "dataset"], errors="ignore", inplace=True)

    # 3. Map string categoricals
    df["sex"]     = df["sex"].map(_SEX_MAP)
    df["cp"]      = df["cp"].str.lower().map(_CP_MAP)
    df["restecg"] = df["restecg"].str.lower().map(_RESTECG_MAP)
    df["slope"]   = df["slope"].str.lower().map(_SLOPE_MAP)
    df["thal"]    = df["thal"].str.lower().map(_THAL_MAP)

    # 4. Map booleans
    df["fbs"]   = df["fbs"].map(_BOOL_MAP)
    df["exang"] = df["exang"].map(_BOOL_MAP)

    # 5. Rename columns to standard names
    df.rename(columns={"thalch": "thalach", "num": "target"}, inplace=True)

    # 6. Binarise target
    df["target"] = (df["target"] > 0).astype(int)

    # 7. Drop missing rows
    rows_before = len(df)
    df.dropna(inplace=True)
    dropped = rows_before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} rows with missing values.")

    # Enforce int types
    int_cols = CATEGORICAL_FEATURES + BINARY_FEATURES + ["target"]
    df[int_cols] = df[int_cols].astype(int)

    df.reset_index(drop=True, inplace=True)
    print(f"[data_loader] Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[data_loader] Class balance — 0 (no disease): {(df['target']==0).sum()}, "
          f"1 (disease): {(df['target']==1).sum()}")
    return df


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (target) as separate objects."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()
    return X, y


def load_and_clean(path: str | Path = RAW_DATA_PATH, dataset: str = "Cleveland") -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    df = load_raw_data(path)
    df = clean_data(df, dataset=dataset)
    return df
