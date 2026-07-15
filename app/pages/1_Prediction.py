"""
Prediction Page — CardioScope-XAI Dashboard

Enter patient clinical data and get a cardiovascular risk score.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction · CardioScope-XAI", page_icon="🔴", layout="wide")

st.title("🔴 Patient Risk Prediction")
st.caption("Enter the patient's clinical measurements to generate a cardiovascular risk score.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────

with st.form("patient_form"):
    st.subheader("Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age (years)", 20, 80, 55)
        sex      = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp       = st.selectbox("Chest Pain Type",
                                options=[1, 2, 3, 4],
                                format_func=lambda x: {
                                    1: "1 — Typical Angina",
                                    2: "2 — Atypical Angina",
                                    3: "3 — Non-Anginal",
                                    4: "4 — Asymptomatic"
                                }[x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)

    with col2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg  = st.selectbox("Resting ECG",
                                options=[0, 1, 2],
                                format_func=lambda x: {
                                    0: "0 — Normal",
                                    1: "1 — ST-T Abnormality",
                                    2: "2 — LV Hypertrophy"
                                }[x])
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang    = st.selectbox("Exercise-Induced Angina",
                                options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope    = st.selectbox("ST Slope",
                                options=[1, 2, 3],
                                format_func=lambda x: {
                                    1: "1 — Upsloping",
                                    2: "2 — Flat",
                                    3: "3 — Downsloping"
                                }[x])
        ca       = st.selectbox("Major Vessels (Ca)", options=[0, 1, 2, 3])
        thal     = st.selectbox("Thalassemia",
                                options=[3, 6, 7],
                                format_func=lambda x: {
                                    3: "3 — Normal",
                                    6: "6 — Fixed Defect",
                                    7: "7 — Reversable Defect"
                                }[x])

    submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────

if submitted:
    clinical_inputs = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    with st.spinner("Running CardioScope-XAI inference..."):
        try:
            from src.predict import predict
            result = predict(clinical_inputs)
            st.session_state["last_result"]  = result
            st.session_state["last_inputs"]  = clinical_inputs
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Make sure all models are trained and saved in the `models/` directory.")
            st.stop()

    import pprint
    output_summary = {
        "probability": result["probability"],
        "risk_tier":   result["risk_tier"],
        "risk_label":  result["risk_label"],
        "xgb_prob":    result["xgb_prob"],
    }
    print("\n" + "="*50)
    print("INPUTS:")
    pprint.pprint(clinical_inputs)
    print("\nOUTPUTS:")
    pprint.pprint(output_summary)
    print("="*50 + "\n")

    st.divider()
    st.subheader("Risk Assessment Result")

    prob      = result["probability"]
    tier      = result["risk_tier"]
    label     = result["risk_label"]
    color     = result["risk_color"]
    xgb_prob  = result["xgb_prob"]

    # ── Risk tier banner
    tier_emoji = {0: "🟢", 1: "🟡", 2: "🔴"}
    st.markdown(
        f"<div style='background:{color}22; border-left:6px solid {color}; "
        f"padding:16px 20px; border-radius:6px; margin-bottom:16px;'>"
        f"<h2 style='color:{color}; margin:0'>{tier_emoji[tier]} {label}</h2>"
        f"<p style='margin:4px 0 0 0; color:#555;'>Fusion model disease probability: "
        f"<b>{prob:.1%}</b></p></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0,  35], "color": "#E8F5E9"},
                    {"range": [35, 65], "color": "#FFF8E1"},
                    {"range": [65, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
            title={"text": "Disease Probability", "font": {"size": 14}},
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Model Scores")
        st.metric("Fusion Model Probability", f"{prob:.1%}")
        st.metric("XGBoost-Only Probability", f"{xgb_prob:.1%}")
        st.divider()
        st.markdown("#### Risk Thresholds")
        st.markdown("🟢 **Low Risk** — probability < 35%")
        st.markdown("🟡 **Medium Risk** — 35% – 65%")
        st.markdown("🔴 **High Risk** — probability > 65%")

    st.divider()
    st.info("Navigate to **Explanation** to see which clinical factors are driving this risk score, "
            "or to **Simulation** to test lifestyle interventions.")
