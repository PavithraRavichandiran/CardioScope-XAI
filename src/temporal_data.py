"""
Synthetic temporal health data generator for CardioScope-XAI.

Generates 12-timestep sequential readings per patient anchored to their
UCI clinical values. High-risk patients trend upward in BP/cholesterol
and downward in heart rate — simulating deteriorating cardiovascular health.

Output shape: (n_patients, 12, 4)
Features per timestep: [systolic_bp, heart_rate, cholesterol, oldpeak]
"""

import numpy as np
import pandas as pd
from pathlib import Path

_ROOT           = Path(__file__).resolve().parents[1]
TEMPORAL_PATH   = _ROOT / "data" / "temporal" / "temporal_sequences.npy"
TEMPORAL_LABELS = _ROOT / "data" / "temporal" / "temporal_labels.npy"
N_TIMESTEPS     = 12
TEMPORAL_FEATURES = ["systolic_bp", "heart_rate", "cholesterol", "oldpeak"]


def _add_trend(
    base: float,
    n_steps: int,
    trend_slope: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a noisy linear trend starting from base value."""
    steps   = np.arange(n_steps)
    trend   = base + trend_slope * steps
    noise   = rng.normal(0, noise_std, n_steps)
    return trend + noise


def generate_temporal_sequences(
    df: pd.DataFrame,
    n_timesteps: int = N_TIMESTEPS,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic longitudinal health sequences from UCI static features.

    For each patient:
      - High-risk (target=1): BP and cholesterol trend upward, heart rate trends down
      - Low-risk  (target=0): values fluctuate near stable baseline

    Args:
        df          : Clean UCI dataframe with 'trestbps', 'thalach', 'chol',
                      'oldpeak', and 'target' columns.
        n_timesteps : Number of monthly readings to simulate per patient.
        random_seed : Reproducibility seed.

    Returns:
        sequences : np.ndarray of shape (n_patients, n_timesteps, 4)
        labels    : np.ndarray of shape (n_patients,)
    """
    rng = np.random.default_rng(random_seed)
    n_patients = len(df)
    sequences  = np.zeros((n_patients, n_timesteps, 4), dtype=np.float32)

    for i, row in df.iterrows():
        is_high_risk = int(row["target"]) == 1

        # Trend slopes: positive = worsening for BP/chol, negative = worsening for HR
        if is_high_risk:
            bp_slope    =  rng.uniform(0.4,  1.2)   # rising BP
            hr_slope    = -rng.uniform(0.3,  0.8)   # declining max HR
            chol_slope  =  rng.uniform(0.5,  1.5)   # rising cholesterol
            op_slope    =  rng.uniform(0.02, 0.08)  # rising ST depression
        else:
            bp_slope    =  rng.uniform(-0.3, 0.3)
            hr_slope    =  rng.uniform(-0.2, 0.2)
            chol_slope  =  rng.uniform(-0.4, 0.4)
            op_slope    =  rng.uniform(-0.01, 0.02)

        # Noise levels based on clinical variability
        bp_seq   = _add_trend(row["trestbps"], n_timesteps, bp_slope,   noise_std=4.0,  rng=rng)
        hr_seq   = _add_trend(row["thalach"],  n_timesteps, hr_slope,   noise_std=3.0,  rng=rng)
        chol_seq = _add_trend(row["chol"],     n_timesteps, chol_slope, noise_std=8.0,  rng=rng)
        op_seq   = _add_trend(row["oldpeak"],  n_timesteps, op_slope,   noise_std=0.15, rng=rng)

        # Clip to physiologically plausible ranges
        bp_seq   = np.clip(bp_seq,   80,   220)
        hr_seq   = np.clip(hr_seq,   40,   210)
        chol_seq = np.clip(chol_seq, 100,  600)
        op_seq   = np.clip(op_seq,   0.0,  6.2)

        sequences[i] = np.stack([bp_seq, hr_seq, chol_seq, op_seq], axis=1)

    labels = df["target"].values.astype(np.int32)
    return sequences, labels


def normalize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Min-max normalize each feature channel across the full dataset.

    Returns normalized sequences and the normalization params (for inference).
    """
    n_features = sequences.shape[2]
    norm_params = {}
    normalized  = sequences.copy()

    for f in range(n_features):
        f_min = sequences[:, :, f].min()
        f_max = sequences[:, :, f].max()
        normalized[:, :, f] = (sequences[:, :, f] - f_min) / (f_max - f_min + 1e-8)
        norm_params[TEMPORAL_FEATURES[f]] = {"min": float(f_min), "max": float(f_max)}

    return normalized.astype(np.float32), norm_params


def save_temporal_data(sequences: np.ndarray, labels: np.ndarray) -> None:
    """Save generated sequences and labels to data/temporal/."""
    TEMPORAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEMPORAL_PATH,   sequences)
    np.save(TEMPORAL_LABELS, labels)
    print(f"[temporal_data] Saved sequences → {TEMPORAL_PATH}  shape: {sequences.shape}")
    print(f"[temporal_data] Saved labels    → {TEMPORAL_LABELS} shape: {labels.shape}")


def load_temporal_data() -> tuple[np.ndarray, np.ndarray]:
    """Load saved temporal sequences and labels."""
    if not TEMPORAL_PATH.exists():
        raise FileNotFoundError(
            f"Temporal data not found at {TEMPORAL_PATH}. Run generate first."
        )
    sequences = np.load(TEMPORAL_PATH)
    labels    = np.load(TEMPORAL_LABELS)
    return sequences, labels


def build_temporal_dataset(df: pd.DataFrame, save: bool = True):
    """
    Full pipeline: generate → normalize → optionally save.

    Returns:
        sequences_norm : normalized sequences (n, 12, 4)
        labels         : target labels (n,)
        norm_params    : per-feature min/max used for normalization
    """
    print("[temporal_data] Generating synthetic temporal sequences...")
    sequences, labels = generate_temporal_sequences(df)
    sequences_norm, norm_params = normalize_sequences(sequences)

    print(f"[temporal_data] Sequence shape : {sequences_norm.shape}")
    print(f"[temporal_data] Class balance  — 0: {(labels==0).sum()}  1: {(labels==1).sum()}")

    if save:
        save_temporal_data(sequences_norm, labels)

    return sequences_norm, labels, norm_params
