"""
SHAP-based explainability for CardioScope-XAI.

Provides per-patient and global feature attribution using TreeExplainer
on the XGBoost model. SHAP values are saved for dashboard use.

Outputs:
    - data/processed/shap_values.npy     : raw SHAP values (n_patients, n_features)
    - data/processed/shap_base_value.npy : base (expected) value
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from xgboost import XGBClassifier

SHAP_VALUES_PATH = Path("data/processed/shap_values.npy")
SHAP_BASE_PATH   = Path("data/processed/shap_base_value.npy")


def build_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the XGBoost model."""
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    save: bool = True,
) -> np.ndarray:
    """
    Compute SHAP values for all patients.

    Args:
        explainer : fitted TreeExplainer
        X         : feature matrix (n_patients, n_features)
        save      : persist to data/processed/

    Returns:
        shap_values : (n_patients, n_features) — positive = increases risk
    """
    shap_values = explainer.shap_values(X)

    # TreeExplainer returns list [class0, class1] for binary — take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if save:
        SHAP_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(SHAP_VALUES_PATH, shap_values)
        np.save(SHAP_BASE_PATH,   np.array([explainer.expected_value
                                            if not isinstance(explainer.expected_value, list)
                                            else explainer.expected_value[1]]))
        print(f"[explainability] SHAP values saved → {SHAP_VALUES_PATH}")

    return shap_values


def load_shap_values() -> tuple[np.ndarray, float]:
    """Load saved SHAP values and base value."""
    if not SHAP_VALUES_PATH.exists():
        raise FileNotFoundError("SHAP values not found. Run compute_shap_values() first.")
    shap_values = np.load(SHAP_VALUES_PATH)
    base_value  = float(np.load(SHAP_BASE_PATH))
    return shap_values, base_value


def plot_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
    plot_type: str = "dot",
) -> None:
    """
    Global SHAP summary plot — shows feature importance and direction.

    Args:
        plot_type: 'dot' (beeswarm) or 'bar' (mean absolute)
    """
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance — Global", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()


def plot_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    patient_index: int,
    max_display: int = 12,
) -> None:
    """
    Per-patient SHAP waterfall plot — shows which features push
    this individual's risk up or down from the base rate.
    """
    shap_values = explainer(X)

    # For binary classifiers, take class 1 values
    if shap_values.values.ndim == 3:
        vals       = shap_values.values[patient_index, :, 1]
        base_val   = shap_values.base_values[patient_index, 1]
        data_row   = shap_values.data[patient_index]
        feat_names = X.columns.tolist()
        expl = shap.Explanation(
            values=vals, base_values=base_val,
            data=data_row, feature_names=feat_names
        )
    else:
        expl = shap_values[patient_index]

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(expl, max_display=max_display, show=False)
    plt.title(f"Patient {patient_index} — SHAP Risk Breakdown", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
) -> None:
    """Global bar chart of mean absolute SHAP values (feature importance)."""
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_shap)))
    mean_shap.tail(max_display).plot(kind="barh", ax=ax, color=colors[-max_display:],
                                      edgecolor="white")
    ax.set_title("Mean |SHAP| — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.show()


def get_patient_explanation(
    shap_values: np.ndarray,
    base_value: float,
    X: pd.DataFrame,
    patient_index: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return a clean DataFrame of the top N SHAP contributors for one patient.
    Positive SHAP = increases risk. Negative = reduces risk.

    Used by the Streamlit dashboard to render text explanations.
    """
    patient_shap = shap_values[patient_index]
    patient_vals = X.iloc[patient_index]

    explanation = pd.DataFrame({
        "feature":    X.columns,
        "value":      patient_vals.values,
        "shap":       patient_shap,
        "direction":  ["↑ Risk" if s > 0 else "↓ Risk" for s in patient_shap],
    })
    explanation["abs_shap"] = explanation["shap"].abs()
    explanation = explanation.sort_values("abs_shap", ascending=False).head(top_n)
    explanation = explanation.drop(columns="abs_shap").reset_index(drop=True)

    return explanation
