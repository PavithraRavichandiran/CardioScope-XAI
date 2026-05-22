"""
UCI Cleveland Heart Disease dataset loader.

Expected file: data/raw/heart.csv
Source: https://archive.ics.uci.edu/ml/datasets/heart+Disease

Columns (13 clinical features + target):
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal, target
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Column definitions ────────────────────────────────────────────────────────

UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Features the model will train on (target excluded)
FEATURE_COLUMNS = UCI_COLUMNS[:-1]

# Categorical features that need encoding
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Continuous features
CONTINUOUS_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# Binary features (already 0/1)
BINARY_FEATURES = ["sex", "fbs", "exang", "ca"]

RAW_DATA_PATH = Path("data/raw/heart.csv")


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the UCI Cleveland dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download heart.csv from:\n"
            "  https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci\n"
            "and place it in data/raw/heart.csv"
        )

    df = pd.read_csv(path)

    # Accept files with or without headers
    if df.columns.tolist() == UCI_COLUMNS:
        pass  # already has correct headers
    elif df.shape[1] == 14:
        df.columns = UCI_COLUMNS
    else:
        raise ValueError(
            f"Unexpected column count: {df.shape[1]}. Expected 14 (13 features + target)."
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw UCI dataset.

    - Replace '?' placeholders with NaN
    - Cast columns to correct dtypes
    - Drop rows with missing values (only 6 rows in UCI Cleveland)
    - Binarise target: 0 = no disease, 1 = disease (original values 1–4)
    """
    df = df.copy()

    # Replace '?' (present in some raw UCI versions) with NaN
    df.replace("?", np.nan, inplace=True)

    # Cast to numeric (coerce any remaining non-numeric to NaN)
    for col in UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop the ~6 rows with missing values
    rows_before = len(df)
    df.dropna(inplace=True)
    dropped = rows_before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} rows with missing values.")

    # Binarise target: values 1,2,3,4 → 1 (disease present)
    df["target"] = (df["target"] > 0).astype(int)

    # Enforce integer types for categorical/binary columns
    int_cols = CATEGORICAL_FEATURES + BINARY_FEATURES + ["target"]
    df[int_cols] = df[int_cols].astype(int)

    df.reset_index(drop=True, inplace=True)
    print(f"[data_loader] Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[data_loader] Class distribution — 0 (no disease): {(df['target']==0).sum()}, "
          f"1 (disease): {(df['target']==1).sum()}")
    return df


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (target) as separate DataFrames."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()
    return X, y


def load_and_clean(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    df = load_raw_data(path)
    df = clean_data(df)
    return df
