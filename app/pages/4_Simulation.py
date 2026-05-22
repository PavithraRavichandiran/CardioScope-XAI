"""
Simulation Page — CardioScope-XAI Dashboard

What-if lifestyle intervention simulator.
Adjust clinical values and see how the risk score changes in real time.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Simulation · CardioScope-XAI", page_icon="🎛️", layout="wide")

st.title("🎛️ What-If Lifestyle Simulation")
st.caption("Adjust clinical parameters to simulate the impact of lifestyle interventions on cardiovascular risk.")
st.divider()

if "last_inputs" not in st.session_state:
    st.warning("No patient data found. Please run a prediction on the **Prediction** page first.")
    st.stop()

base_inputs = st.session_state["last_inputs"]
base_result = st.session_state["last_result"]
base_prob   = base_result["probability"]
base_label  = base_result["risk_label"]
base_color  = base_result["risk_color"]

# ── Baseline display ──────────────────────────────────────────────────────────

st.markdown(
    f"<div style='background:{base_color}22; border-left:6px solid {base_color}; "
    f"padding:12px 18px; border-radius:6px; margin-bottom:16px;'>"
    f"<b>Baseline Risk:</b> {base_label} &nbsp;|&nbsp; "
    f"<b>Probability:</b> {base_prob:.1%}</div>",
    unsafe_allow_html=True,
)

st.subheader("Adjust Clinical Parameters")
st.caption("Modify any value below to simulate the effect of a clinical intervention or lifestyle change.")

# ── Intervention sliders ──────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Modifiable Risk Factors**")
    new_trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        80, 200,
        int(base_inputs["trestbps"]),
        help="Lowering BP through medication or exercise reduces cardiovascular risk.",
    )
    new_chol = st.slider(
        "Cholesterol (mg/dl)",
        100, 600,
        int(base_inputs["chol"]),
        help="Diet, statins, and exercise can reduce cholesterol levels.",
    )
    new_thalach = st.slider(
        "Max Heart Rate Achieved",
        60, 220,
        int(base_inputs["thalach"]),
        help="Higher max HR achieved suggests better cardiovascular fitness.",
    )

with col2:
    st.markdown("**Lifestyle & Clinical Changes**")
    new_oldpeak = st.slider(
        "ST Depression (oldpeak)",
        0.0, 6.2,
        float(base_inputs["oldpeak"]),
        step=0.1,
        help="Lower ST depression after exercise indicates healthier cardiac response.",
    )
    new_exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        index=int(base_inputs["exang"]),
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Treating angina may reduce this flag from Yes to No.",
    )
    new_fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        index=int(base_inputs["fbs"]),
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Managing diabetes can bring fasting blood sugar under control.",
    )

run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

# ── Run simulation ────────────────────────────────────────────────────────────

if run_simulation:
    modifications = {
        "trestbps": new_trestbps,
        "chol":     new_chol,
        "thalach":  new_thalach,
        "oldpeak":  new_oldpeak,
        "exang":    new_exang,
        "fbs":      new_fbs,
    }

    with st.spinner("Running what-if simulation..."):
        try:
            from src.predict import simulate_what_if
            sim_result = simulate_what_if(base_inputs, modifications)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    st.divider()
    st.subheader("Simulation Result")

    new_prob  = sim_result["probability"]
    new_label = sim_result["risk_label"]
    new_color = sim_result["risk_color"]
    delta     = sim_result["delta_probability"]

    # ── Side-by-side comparison ───────────────────────────────────────────────

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(f"**Baseline**")
        st.markdown(
            f"<div style='background:{base_color}22; border-left:5px solid {base_color}; "
            f"padding:12px; border-radius:6px;'>"
            f"<h3 style='color:{base_color}; margin:0'>{base_label}</h3>"
            f"<p style='margin:4px 0 0 0'>{base_prob:.1%}</p></div>",
            unsafe_allow_html=True,
        )

    with col2:
        arrow = "⬇️" if delta < 0 else ("⬆️" if delta > 0 else "➡️")
        delta_color = "#4CAF50" if delta < 0 else ("#F44336" if delta > 0 else "#888")
        st.markdown(f"<div style='text-align:center; padding-top:30px;'>"
                    f"<h2>{arrow}</h2>"
                    f"<p style='color:{delta_color}; font-weight:bold'>{delta:+.1%}</p>"
                    f"</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"**After Intervention**")
        st.markdown(
            f"<div style='background:{new_color}22; border-left:5px solid {new_color}; "
            f"padding:12px; border-radius:6px;'>"
            f"<h3 style='color:{new_color}; margin:0'>{new_label}</h3>"
            f"<p style='margin:4px 0 0 0'>{new_prob:.1%}</p></div>",
            unsafe_allow_html=True,
        )

    # ── Gauge comparison ──────────────────────────────────────────────────────

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=base_prob * 100,
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": "Baseline", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": base_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35], "color": "#E8F5E9"},
                {"range": [35, 65], "color": "#FFF8E1"},
                {"range": [65, 100], "color": "#FFEBEE"},
            ],
        },
        domain={"x": [0, 0.45], "y": [0, 1]},
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=new_prob * 100,
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": "After Intervention", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": new_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35], "color": "#E8F5E9"},
                {"range": [35, 65], "color": "#FFF8E1"},
                {"range": [65, 100], "color": "#FFEBEE"},
            ],
        },
        domain={"x": [0.55, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── What changed ──────────────────────────────────────────────────────────

    st.divider()
    st.subheader("What Changed")

    changes = []
    labels_map = {
        "trestbps": "Resting BP", "chol": "Cholesterol",
        "thalach": "Max Heart Rate", "oldpeak": "ST Depression",
        "exang": "Exercise Angina", "fbs": "Fasting Blood Sugar",
    }
    for feat, new_val in modifications.items():
        old_val = base_inputs[feat]
        if old_val != new_val:
            changes.append({
                "Parameter":   labels_map.get(feat, feat),
                "Before":      old_val,
                "After":       new_val,
                "Change":      f"{new_val - old_val:+.1f}" if isinstance(new_val, float)
                               else f"{int(new_val) - int(old_val):+d}",
            })

    if changes:
        st.dataframe(changes, use_container_width=True, hide_index=True)
    else:
        st.info("No parameters were changed from baseline.")

    if delta < -0.05:
        st.success(f"These interventions reduced cardiovascular risk by **{abs(delta):.1%}**. "
                   f"Focus on the highest-impact changes identified above.")
    elif delta > 0.05:
        st.error(f"These changes increased cardiovascular risk by **{delta:.1%}**.")
    else:
        st.info("Minimal change in risk. Try more significant interventions.")
