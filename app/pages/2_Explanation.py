"""
Explanation Page — CardioScope-XAI Dashboard

Shows SHAP-based per-patient risk factor breakdown.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Explanation · CardioScope-XAI", page_icon="🔍", layout="wide")

st.title("🔍 Risk Factor Explanation")
st.caption("SHAP-based breakdown of which clinical factors are driving this patient's risk score.")
st.divider()

# ── Require prediction first ──────────────────────────────────────────────────

if "last_result" not in st.session_state:
    st.warning("No prediction found. Please run a prediction on the **Prediction** page first.")
    st.stop()

result  = st.session_state["last_result"]
inputs  = st.session_state["last_inputs"]
shap_df = result["shap_df"]
prob    = result["probability"]
label   = result["risk_label"]
color   = result["risk_color"]

# ── Summary banner ────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='background:{color}22; border-left:6px solid {color}; "
    f"padding:12px 18px; border-radius:6px; margin-bottom:16px;'>"
    f"<b>Current Risk:</b> {label} &nbsp;|&nbsp; <b>Probability:</b> {prob:.1%}</div>",
    unsafe_allow_html=True,
)

st.subheader("Top Risk Drivers for This Patient")
st.caption("Positive SHAP = increases disease risk.  Negative SHAP = reduces disease risk.")

# ── SHAP waterfall (Plotly) ───────────────────────────────────────────────────

shap_df_sorted = shap_df.sort_values("shap")
bar_colors = ["#F44336" if v > 0 else "#4CAF50" for v in shap_df_sorted["shap"]]

fig = go.Figure(go.Bar(
    x=shap_df_sorted["shap"],
    y=[f"{row['feature']} = {row['value']:.2f}" for _, row in shap_df_sorted.iterrows()],
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:+.4f}" for v in shap_df_sorted["shap"]],
    textposition="outside",
))

fig.update_layout(
    title="SHAP Feature Contributions (Top 5 Drivers)",
    xaxis_title="SHAP Value (impact on disease probability)",
    yaxis_title="",
    height=360,
    margin=dict(l=20, r=60, t=50, b=40),
    xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1.5),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Tabular explanation ───────────────────────────────────────────────────────

st.subheader("Detailed Breakdown")

FEATURE_DESCRIPTIONS = {
    "age":           "Patient age in years",
    "trestbps":      "Resting blood pressure (mm Hg)",
    "chol":          "Serum cholesterol (mg/dl)",
    "thalach":       "Maximum heart rate achieved",
    "oldpeak":       "ST depression induced by exercise",
    "ca":            "Number of major vessels coloured by fluoroscopy",
    "map_value":     "Mean arterial pressure (derived)",
    "chol_hr_ratio": "Cholesterol to max heart rate ratio (derived)",
    "hr_reserve":    "Heart rate reserve vs age-predicted max (derived)",
    "st_risk_index": "ST depression × exercise angina (derived)",
    "age_bp_index":  "Age × blood pressure load (derived)",
}

rows = []
for _, row in shap_df.iterrows():
    direction_icon = "🔴 ↑" if row["shap"] > 0 else "🟢 ↓"
    rows.append({
        "Feature":      row["feature"],
        "Value":        f"{row['value']:.3f}",
        "SHAP Impact":  f"{row['shap']:+.4f}",
        "Direction":    direction_icon + " Risk",
        "Description":  FEATURE_DESCRIPTIONS.get(row["feature"], "—"),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── Clinical insight ──────────────────────────────────────────────────────────

st.subheader("Clinical Interpretation")

top_risk_driver = shap_df.loc[shap_df["shap"].idxmax(), "feature"]
top_protective  = shap_df.loc[shap_df["shap"].idxmin(), "feature"]

col1, col2 = st.columns(2)
with col1:
    st.error(f"**Highest Risk Factor:** `{top_risk_driver}`\n\n"
             f"This feature is pushing this patient's risk **upward** the most. "
             f"Consider clinical attention to this parameter.")
with col2:
    st.success(f"**Strongest Protective Factor:** `{top_protective}`\n\n"
               f"This feature is reducing this patient's risk the most. "
               f"Maintaining or improving it may lower overall risk.")

st.info("Navigate to **Simulation** to see how changing clinical values affects the risk score.")
