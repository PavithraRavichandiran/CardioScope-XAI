# 🫀 CardioScope-XAI

> A hybrid LSTM–XGBoost explainable AI framework for early and interpretable heart disease risk prediction using temporal and clinical data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Tabular_Classifier-orange?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)

---

## 🚨 Problem

Cardiovascular disease is the **#1 cause of death globally** — yet most patients are identified as high-risk only *after* a cardiac event has already occurred.

Existing prediction systems have three critical failures:

- **Static snapshots only** — they look at one point in time, missing how a patient's health trends over weeks and months
- **Binary output** — "disease / no disease" gives clinicians no actionable risk level
- **Black-box predictions** — doctors can't trust a model they can't explain

The result: missed early diagnoses, delayed treatment, and preventable deaths.

---

## 💡 Our Approach

CardioScope-XAI solves this with a **three-layer architecture**:

```
Sequential Health Data          Static Clinical Data
(BP trends, HR, lipids)         (age, cholesterol, ECG...)
        │                               │
        ▼                               ▼
   LSTM Network               XGBoost Classifier
   (temporal patterns)        (tabular risk scoring)
        │                               │
        └──────────── FUSION ───────────┘
                          │
                          ▼
              Unified Risk Score
           (Low / Medium / High)
                          │
                          ▼
              SHAP Explainability
          (why is this patient at risk?)
                          │
                          ▼
           Interactive Streamlit Dashboard
         (predict → explain → simulate)
```

### Layer 1 — LSTM (Temporal Intelligence)
Learns *how a patient's health is changing over time* — not just where it is today. Rising BP trend over 6 months signals higher risk than stable high BP.

### Layer 2 — XGBoost (Clinical Intelligence)
High-accuracy classification on 13 static clinical features from the UCI Cleveland dataset. Evaluated against 10 algorithms — XGBoost achieved **98.6% accuracy**.

### Layer 3 — Feature-Level Fusion
LSTM latent state representations are concatenated with XGBoost features to produce a single unified cardiovascular risk score — combining the best of both worlds.

---

## 🔍 Explainability (SHAP)

Most medical AI is a black box. Doctors won't use what they can't understand.

CardioScope-XAI uses **SHAP (SHapley Additive Explanations)** to show:
- Which clinical features are driving **each individual patient's** risk score
- How much each factor (age, cholesterol, ST depression, etc.) contributes
- What would change if a patient improved their lifestyle (what-if simulation)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔴 Risk Tiers | Low / Medium / High — not just disease/no disease |
| 📈 Temporal Analysis | LSTM learns health trends over time |
| 🤖 Hybrid Model | LSTM + XGBoost feature-level fusion |
| 🔎 Per-Patient SHAP | Explainability for every prediction |
| 🎛️ What-If Simulation | "What if my BP drops by 10?" — live risk update |
| 📊 Clinical Dashboard | Streamlit app for non-technical healthcare staff |
| 🧪 MLflow Tracking | Full experiment reproducibility |

---

## 🗂️ Project Structure

```
CardioScope-XAI/
│
├── data/
│   ├── raw/                  # Original UCI dataset
│   ├── temporal/             # Sequential/wearable data
│   └── processed/            # Cleaned, merged, ready-to-train
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_lstm_model.ipynb
│   ├── 05_fusion.ipynb
│   └── 06_shap_explainability.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── fusion.py
│   ├── explainability.py
│   └── predict.py
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── prediction.py
│       ├── explanation.py
│       ├── trends.py
│       └── simulation.py
│
├── models/                   # Saved model artifacts
├── mlflow/                   # Experiment tracking
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Tabular | XGBoost, scikit-learn |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| Dataset | UCI Cleveland Heart Disease (303 records, 13 features) |

---

## 📊 Dataset

**Primary:** [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) — 303 patient records, 13 clinical features

**Temporal Component:** Sequential physiological readings (blood pressure, heart rate, lipid levels over time) — simulated using clinical statistical distributions to train the LSTM module.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/CardioScope-XAI.git
cd CardioScope-XAI

# Install dependencies
pip install -r requirements.txt
```

**Place the dataset:**
Download `heart_disease_uci.csv` from Kaggle and place it at `data/raw/heart_disease_uci.csv`.

**Run notebooks in order:**
```bash
jupyter notebook notebooks/01_EDA.ipynb              # Explore data
jupyter notebook notebooks/02_preprocessing.ipynb    # Engineer features + save scaler
jupyter notebook notebooks/03_xgboost_model.ipynb   # Train XGBoost + log to MLflow
jupyter notebook notebooks/04_lstm_model.ipynb       # Train LSTM + save embeddings
jupyter notebook notebooks/05_fusion.ipynb           # Train fusion classifier
jupyter notebook notebooks/06_shap_explainability.ipynb  # Compute SHAP values
```

**Launch the dashboard:**
```bash
streamlit run app/app.py
```

---

## 👥 Contributors

| Name | Role |
|---|---|
| Pavithra Ravichandiran | ML Engineering, Backend, Deployment |
| [Collaborator Name] | [Their role] |

---

## 📌 Project Status

✅ **Complete** — All core modules and dashboard built

- [x] Problem Statement
- [x] Repository Setup
- [x] EDA (`notebooks/01_EDA.ipynb`)
- [x] Preprocessing (`notebooks/02_preprocessing.ipynb`)
- [x] XGBoost Model + MLflow (`notebooks/03_xgboost_model.ipynb`)
- [x] LSTM Model + Temporal Data (`notebooks/04_lstm_model.ipynb`)
- [x] Fusion Layer (`notebooks/05_fusion.ipynb`)
- [x] SHAP Explainability (`notebooks/06_shap_explainability.ipynb`)
- [x] Streamlit Dashboard — Prediction, Explanation, Trends, Simulation
- [ ] LinkedIn Post

---

## 📄 License

MIT License — open for research and educational use.

---

<p align="center">Built with ❤️ for preventive cardiology · CardioScope-XAI · 2026</p>
