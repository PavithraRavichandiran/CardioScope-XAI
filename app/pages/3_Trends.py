"""
Trends Page — CardioScope-XAI Dashboard

Visualises the patient's simulated longitudinal health trends
used by the LSTM temporal model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Trends · CardioScope-XAI", page_icon="📈", layout="wide")

st.title("📈 Temporal Health Trends")
st.caption("12-month longitudinal health profile used by the LSTM temporal model.")
st.divider()

if "last_inputs" not in st.session_state:
    st.warning("No patient data found. Please run a prediction on the **Prediction** page first.")
    st.stop()

inputs = st.session_state["last_inputs"]
result = st.session_state["last_result"]
prob   = result["probability"]
color  = result["risk_color"]
label  = result["risk_label"]

# ── Generate the temporal sequence for this patient ───────────────────────────

with st.spinner("Generating temporal health sequence..."):
    try:
        import pandas as pd
        from src.predict import make_patient_df, _load_models, _generate_sequence_for_patient
        from src.temporal_data import TEMPORAL_FEATURES

        raw_df   = make_patient_df(inputs)
        raw_df["target"] = 0
        sequence = _generate_sequence_for_patient(raw_df, inputs)  # (12, 4)
    except Exception as e:
        st.error(f"Could not generate temporal sequence: {e}")
        st.stop()

months     = [f"Month {i+1}" for i in range(sequence.shape[0])]
feat_labels = {
    "systolic_bp":  "Systolic Blood Pressure (normalised)",
    "heart_rate":   "Heart Rate (normalised)",
    "cholesterol":  "Cholesterol (normalised)",
    "oldpeak":      "ST Depression (normalised)",
}
feat_colors = ["#1565C0", "#C62828", "#6A1B9A", "#E65100"]

# ── Risk banner ───────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='background:{color}22; border-left:6px solid {color}; "
    f"padding:12px 18px; border-radius:6px; margin-bottom:16px;'>"
    f"<b>Patient Risk:</b> {label} &nbsp;|&nbsp; "
    f"<b>Fusion Probability:</b> {prob:.1%}</div>",
    unsafe_allow_html=True,
)

# ── 4-panel trend chart ───────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(feat_labels.values()),
    vertical_spacing=0.15,
    horizontal_spacing=0.08,
)

positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for i, (feat, clr) in enumerate(zip(TEMPORAL_FEATURES, feat_colors)):
    row, col = positions[i]
    values   = sequence[:, i]

    # Trend line
    fig.add_trace(go.Scatter(
        x=months, y=values,
        mode="lines+markers",
        name=feat_labels[feat],
        line=dict(color=clr, width=2.5),
        marker=dict(size=6, color=clr),
        showlegend=False,
    ), row=row, col=col)

    # Shaded area under curve
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=list(values) + [0] * len(months),
        fill="toself",
        fillcolor=clr + "18",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ), row=row, col=col)

    # Trend annotation
    slope = np.polyfit(range(len(values)), values, 1)[0]
    trend_text = "↑ Rising" if slope > 0.005 else ("↓ Falling" if slope < -0.005 else "→ Stable")
    trend_color = "#F44336" if slope > 0.005 else ("#4CAF50" if slope < -0.005 else "#888")

    fig.add_annotation(
        text=f"<b>{trend_text}</b>",
        x=months[-1], y=values[-1],
        xref=f"x{'' if i==0 else i+1}", yref=f"y{'' if i==0 else i+1}",
        showarrow=False,
        font=dict(size=11, color=trend_color),
        xanchor="right",
    )

fig.update_layout(
    height=520,
    title_text="12-Month Physiological Trends (LSTM Input)",
    title_font_size=14,
    margin=dict(t=60, b=20, l=20, r=20),
    plot_bgcolor="white",
)
fig.update_xaxes(tickangle=45, tickfont_size=9)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Trend summary table ───────────────────────────────────────────────────────

st.subheader("Trend Summary")

summary_rows = []
for i, feat in enumerate(TEMPORAL_FEATURES):
    values = sequence[:, i]
    slope  = np.polyfit(range(len(values)), values, 1)[0]
    trend  = "↑ Rising" if slope > 0.005 else ("↓ Falling" if slope < -0.005 else "→ Stable")
    concern = (feat in ["systolic_bp", "cholesterol", "oldpeak"] and slope > 0.005) or \
              (feat == "heart_rate" and slope < -0.005)
    summary_rows.append({
        "Feature":     feat_labels[feat],
        "Start":       f"{values[0]:.3f}",
        "End":         f"{values[-1]:.3f}",
        "Trend":       trend,
        "Clinical Flag": "⚠️ Worsening" if concern else "✅ OK",
    })

st.dataframe(summary_rows, use_container_width=True, hide_index=True)

st.divider()
st.caption("Trends are simulated from UCI static values using clinical statistical distributions. "
           "Rising BP / cholesterol and falling HR are associated with increased cardiovascular risk.")
