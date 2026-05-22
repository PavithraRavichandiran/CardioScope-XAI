"""
CardioScope-XAI — Main Streamlit App Entry Point

Run with:
    streamlit run app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CardioScope-XAI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
st.sidebar.title("CardioScope-XAI")
st.sidebar.caption("Hybrid LSTM–XGBoost Explainable AI\nfor Heart Disease Risk Prediction")
st.sidebar.divider()
st.sidebar.markdown("""
**Navigate using the pages above.**

| Page | Purpose |
|---|---|
| 🔴 Prediction | Enter patient data → risk score |
| 🔍 Explanation | SHAP breakdown of risk drivers |
| 📈 Trends | Temporal health trends (LSTM) |
| 🎛️ Simulation | What-if lifestyle changes |
""")
st.sidebar.divider()
st.sidebar.caption("CardioScope-XAI · 2026 · For research use only")

# ── Home page ─────────────────────────────────────────────────────────────────

st.title("🫀 CardioScope-XAI")
st.subheader("A Hybrid LSTM–XGBoost Explainable AI Framework for Heart Disease Risk Prediction")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "LSTM + XGBoost")
with col2:
    st.metric("Dataset", "UCI Cleveland")
with col3:
    st.metric("Explainability", "SHAP")
with col4:
    st.metric("Risk Tiers", "Low / Med / High")

st.divider()

st.markdown("""
### How it works

```
Patient Data (13 clinical features)       Sequential Health Trends (12 months)
         │                                              │
         ▼                                              ▼
  XGBoost Classifier                           LSTM Encoder
  (tabular risk scoring)                  (temporal pattern learning)
         │                                              │
         └──────────────── FUSION ────────────────────┘
                                │
                                ▼
                    Unified Cardiovascular Risk Score
                        Low  /  Medium  /  High
                                │
                                ▼
                     SHAP Explainability
               (which factors drive this patient's risk?)
```

### Get Started
Use the sidebar to navigate to **Prediction** and enter a patient's clinical data.
""")

st.divider()
st.caption("For clinical research and educational use only. Not a substitute for medical diagnosis.")
