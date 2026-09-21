"""
RiskPulse — Heart Disease Risk Intelligence
==========================================
An explainable machine-learning decision-support tool for heart disease
risk prediction, built with Streamlit.

Author: Ajeboriogbon Samuel A. (pka SBM ~ Nucopia)
Dept.  : Statistics, Obafemi Awolowo University, Ile-Ife, Nigeria
Model : Pre-trained classifier (model.pkl) + StandardScaler (scaler.pkl)
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="RiskPulse | Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. THEME / CSS
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background:
      radial-gradient(1100px 600px at 12% -10%, rgba(244,114,182,.14), transparent 60%),
      radial-gradient(900px 600px at 90% 0%, rgba(129,140,248,.20), transparent 55%),
      linear-gradient(180deg, #070b14 0%, #0b1220 45%, #070b14 100%);
    color: #e6edf7;
}
#MainMenu, footer { visibility: hidden; }

/* ---------- Hero ---------- */
.hero {
    border-radius: 24px; padding: 30px 34px;
    background: linear-gradient(135deg, rgba(244,114,182,.20), rgba(99,102,241,.22));
    border: 1px solid rgba(148,163,184,.25);
    box-shadow: 0 24px 60px -30px rgba(244,114,182,.55);
    margin-bottom: 22px;
}
.hero h1 {
    font-size: 2.15rem; font-weight: 800; margin: 0 0 8px 0; line-height: 1.15;
    background: linear-gradient(90deg,#f0abfc,#a5b4fc 55%,#5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color:#a8b6cc; font-size:1rem; margin:0; max-width: 68rem; line-height: 1.55;}
.pill {
    display:inline-block; padding:5px 13px; margin:8px 8px 0 0;
    border-radius:999px; font-size:.76rem; font-weight:600; letter-spacing:.3px;
    background: rgba(148,163,184,.13); border:1px solid rgba(148,163,184,.28);
    color:#cbd5e1;
}

/* ---------- Cards ---------- */
.card {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px; padding: 18px 20px; height: 100%;
    backdrop-filter: blur(9px);
}
.kpi-label { color:#93a4bd; font-size:.74rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:700;}
.kpi-value { font-size:1.85rem; font-weight:800; color:#f1f5f9; line-height:1.15; margin-top:4px;}
.kpi-sub   { color:#7e8ea6; font-size:.78rem; margin-top:2px;}

/* ---------- Verdict ---------- */
.verdict {
    border-radius:20px; padding:22px 24px; text-align:center;
    border:1px solid rgba(255,255,255,.14);
}
.verdict h2 { margin:0; font-size:1.55rem; font-weight:800; }
.verdict p  { margin:6px 0 0 0; color:rgba(255,255,255,.82); font-size:.9rem;}

/* ---------- Driver cards ---------- */
.driver-up   { border-left: 4px solid #ef4444; background: rgba(239,68,68,.08);
               padding:.85rem 1rem; border-radius:10px; margin-bottom:.5rem; }
.driver-down { border-left: 4px solid #22c55e; background: rgba(34,197,94,.08);
               padding:.85rem 1rem; border-radius:10px; margin-bottom:.5rem; }
.driver-title { font-weight: 700; color:#e2e8f0; font-size:.95rem; margin-top:2px;}
.driver-tag   { font-size:.72rem; color:#94a3b8; font-weight:600; letter-spacing:.4px;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080d18,#0d1526);
    border-right:1px solid rgba(148,163,184,.16);
}
section[data-testid="stSidebar"] * { color: #dbe3ee; }

.sb-card {
    background: rgba(15,23,42,.55);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(148,163,184,.08); border-radius: 12px 12px 0 0;
    padding: 10px 18px; font-weight:600; color:#cbd5e1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(244,114,182,.28), rgba(129,140,248,.28)) !important;
    color:#ffffff !important;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    width:100%; border-radius:12px; font-weight:700; padding:.75rem 1rem;
    background: linear-gradient(135deg,#ec4899,#6366f1); color:white; border:0;
    box-shadow:0 12px 30px -14px rgba(236,72,153,.9);
    font-size: 1rem;
}
div.stButton > button:hover { filter:brightness(1.08); color:white; transform:translateY(-1px); }

/* ---------- Misc ---------- */
.small { color:#8698b1; font-size:.8rem; }
hr { border-color: rgba(148,163,184,.16) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# 3. MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_asset():
    """Load the pre-trained model and scaler once and reuse them."""
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_asset()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    LOAD_ERROR = str(e)

# =============================================================================
# 4. CONSTANTS — real values from test-set evaluation
# =============================================================================
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------------
# Model metrics — real values from the test-set evaluation
# NOTE: Test set = 61 samples. Metrics carry wide confidence intervals.
#       This is a small holdout; treat as indicative, not definitive.
# ---------------------------------------------------------------------
MODEL_METRICS = {
    "accuracy":     "85%",     # (31 + 30 correct) / 61
    "precision_0":  "0.87",    # No-disease class
    "precision_1":  "0.84",    # Disease class
    "recall_0":     "0.84",    # No-disease class
    "recall_1":     "0.87",    # Disease class — sensitivity for positives
    "f1_macro":     "0.85",
    "test_n":       "61",
}

# ---------------------------------------------------------------------
# Feature importances — from model.feature_importances_ (Random Forest)
# Sorted descending; includes all 13 features for completeness.
# ---------------------------------------------------------------------
FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": [
        "Thalassemia",
        "Chest Pain Type",
        "ST Depression (oldpeak)",
        "Max Heart Rate (thalach)",
        "Major Vessels (ca)",
        "Exercise Angina (exang)",
        "Slope of ST Segment",
        "Age",
        "Serum Cholesterol",
        "Resting Blood Pressure",
        "Sex",
        "Resting ECG",
        "Fasting Blood Sugar",
    ],
    "Importance": [
        0.170256,  # Thalassemia
        0.129720,  # Chest Pain Type
        0.121443,  # ST Depression
        0.108584,  # Max Heart Rate
        0.103718,  # Major Vessels
        0.080626,  # Exercise Angina
        0.074601,  # Slope ST Segment
        0.061227,  # Age
        0.053104,  # Serum Cholesterol
        0.041181,  # Resting BP
        0.031107,  # Sex
        0.020048,  # Resting ECG
        0.004387,  # Fasting Blood Sugar
    ],
})

# =============================================================================
# 5. RISK DRIVERS (rule-based, aligned with real feature importances)
# =============================================================================
def compute_risk_drivers(row: dict) -> list:
    """Lightweight rule-based driver detection. Rules are aligned with the
    Random Forest's actual top-6 features."""
    out = []

    age = row.get("age", 50)
    if age >= 60:
        out.append({"factor": f"Advanced age ({age} yrs)", "impact": "High",
                    "direction": "up"})
    elif age >= 50:
        out.append({"factor": f"Age above 50 ({age} yrs)", "impact": "Moderate",
                    "direction": "up"})

    sex = row.get("sex", 0)
    if sex == 1:
        out.append({"factor": "Male sex", "impact": "Moderate",
                    "direction": "up"})

    cp = row.get("cp", 0)
    if cp == 3:
        out.append({"factor": "Asymptomatic chest pain", "impact": "High",
                    "direction": "up"})
    elif cp == 0:
        out.append({"factor": "Typical angina", "impact": "Moderate",
                    "direction": "up"})

    trestbps = row.get("trestbps", 120)
    if trestbps >= 160:
        out.append({"factor": f"High resting BP ({trestbps} mm Hg)", "impact": "High",
                    "direction": "up"})
    elif trestbps >= 140:
        out.append({"factor": f"Elevated resting BP ({trestbps} mm Hg)", "impact": "Moderate",
                    "direction": "up"})

    chol = row.get("chol", 200)
    if chol >= 300:
        out.append({"factor": f"High cholesterol ({chol} mg/dl)", "impact": "High",
                    "direction": "up"})
    elif chol >= 240:
        out.append({"factor": f"Borderline cholesterol ({chol} mg/dl)", "impact": "Moderate",
                    "direction": "up"})

    thalach = row.get("thalach", 150)
    if thalach >= 160:
        out.append({"factor": f"High max heart rate ({thalach} bpm)", "impact": "Protective",
                    "direction": "down"})
    elif thalach <= 100:
        out.append({"factor": f"Low max heart rate ({thalach} bpm)", "impact": "High",
                    "direction": "up"})

    exang = row.get("exang", 0)
    if exang == 1:
        out.append({"factor": "Exercise-induced angina", "impact": "High",
                    "direction": "up"})

    oldpeak = row.get("oldpeak", 0.0)
    if oldpeak >= 2.0:
        out.append({"factor": f"Significant ST depression ({oldpeak})", "impact": "High",
                    "direction": "up"})
    elif oldpeak >= 1.0:
        out.append({"factor": f"Moderate ST depression ({oldpeak})", "impact": "Moderate",
                    "direction": "up"})

    ca = row.get("ca", 0)
    if ca >= 2:
        out.append({"factor": f"{ca} major vessels affected", "impact": "Very High",
                    "direction": "up"})

    thal = row.get("thal", 0)
    if thal == 3:
        out.append({"factor": "Reversible thalassemia defect", "impact": "High",
                    "direction": "up"})
    elif thal == 2:
        out.append({"factor": "Fixed thalassemia defect", "impact": "Moderate",
                    "direction": "up"})

    out.sort(key=lambda d: 0 if d["direction"] == "up" else 1)
    return out[:6]

# =============================================================================
# 6. SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding-top:8px'>"
        "<div style='font-size:46px;line-height:1'>🫀</div>"
        "<div style='font-weight:800;font-size:1.25rem;color:#e2e8f0;margin-top:4px'>RiskPulse</div>"
        "<div style='color:#8698b1;font-size:.78rem'>Heart Disease Risk Intelligence</div>"
        "</div><hr>",
        unsafe_allow_html=True,
    )

    if MODEL_READY:
        st.success("Model loaded ✓", icon="✅")
    else:
        st.error(f"Model failed to load: {LOAD_ERROR}")

    st.markdown("##### 📌 How to Use")
    st.markdown("""
<div class="sb-card">
  <div class="small" style="line-height:1.6">
  <b>1.</b> Fill in the patient's clinical parameters<br>
  <b>2.</b> Click <b>Assess Health Risk</b><br>
  <b>3.</b> Review the probability gauge<br>
  <b>4.</b> Read the clinical recommendations
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("##### ⚙️ Model Info")
    st.markdown(f"""
<div class="sb-card">
  <div class="small" style="line-height:1.6">
  <b>Algorithm:</b> Random Forest Classifier<br>
  <b>Dataset:</b> UCI Heart Disease<br>
  <b>Features:</b> 13 clinical parameters<br>
  <b>Accuracy:</b> {MODEL_METRICS['accuracy']}<br>
  <b>Sensitivity:</b> {MODEL_METRICS['recall_1']}<br>
  <b>F1 (macro):</b> {MODEL_METRICS['f1_macro']}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("##### 👨‍💻 Developer")
    st.markdown("""
<div class="sb-card">
  <div style="font-weight:800;color:#e2e8f0;font-size:1rem">Samuel Ajeboriogbon</div>
  <div style="color:#f0abfc;font-size:.76rem;font-weight:600;margin-bottom:6px">
    pka SBM ~ Nucopia
  </div>
  <div class="small" style="line-height:1.5">
    🎓 B.Sc. Statistics, OAU Ile-Ife<br>
    💼 Data Scientist &amp; ML Engineer<br>
    🚀 Inspired by TechCrush
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "<div class='small' style='margin-top:14px;line-height:1.5'>"
        "⚠️ <b>Educational tool.</b> Not a substitute for professional "
        "medical diagnosis. Always consult a qualified healthcare provider."
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# 7. HERO
# =============================================================================
st.markdown(
    """
<div class="hero">
  <h1>RiskPulse · Heart Disease Risk Intelligence</h1>
  <p>An explainable machine-learning decision-support tool trained on the
  UCI Heart Disease dataset. Enter a patient's clinical parameters to receive
  an instant risk assessment with actionable clinical guidance.</p>
  <div>
    <span class="pill">🫀 Random Forest</span>
    <span class="pill">📊 13 clinical features</span>
    <span class="pill">⚡ Instant prediction</span>
    <span class="pill">🎓 OAU Ile-Ife · SBM ~ Nucopia</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 8. TABS
# =============================================================================
tab_assess, tab_insights, tab_about = st.tabs(
    ["🩺 Risk Assessment", "📊 Model Insights", "ℹ️ About"]
)

# =============================================================================
# 8.1  RISK ASSESSMENT
# =============================================================================
with tab_assess:
    st.markdown("### 👤 Patient Clinical Parameters")
    st.markdown(
        "<p class='small'>Fill in the fields below. Every parameter maps directly "
        "to a feature the model was trained on.</p>",
        unsafe_allow_html=True,
    )

    # ---------- Input form: 3 columns ----------
    row = {}
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 👤 Demographics")
        row["age"] = st.slider("Age (years)", 20, 100, 50)
        sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
        row["sex"] = 1 if sex_label == "Male" else 0

        st.markdown("##### 💓 Vitals")
        row["trestbps"] = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        row["chol"] = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        row["thalach"] = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)

    with c2:
        st.markdown("##### 🩺 Clinical Markers")
        fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl",
                             ["False", "True"], horizontal=True)
        row["fbs"] = 1 if fbs_label == "True" else 0

        exang_label = st.radio("Exercise-Induced Angina",
                               ["No", "Yes"], horizontal=True)
        row["exang"] = 1 if exang_label == "Yes" else 0

        row["oldpeak"] = st.slider("ST Depression (0.0 – 6.0)", 0.0, 6.0, 1.0, step=0.1)

        cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
        selected_cp = st.selectbox("Chest Pain Type", cp_options)
        row["cp"] = cp_options.index(selected_cp)

    with c3:
        st.markdown("##### 📈 ECG & Imaging")
        restecg_options = ["Normal", "Abnormal", "Structural Issues"]
        selected_restecg = st.selectbox("Resting ECG Results", restecg_options)
        row["restecg"] = restecg_options.index(selected_restecg)

        row["slope"] = st.selectbox("Slope of ST Segment", [0, 1, 2],
                                    help="0 = upsloping, 1 = flat, 2 = downsloping")
        row["ca"] = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
        row["thal"] = st.selectbox("Thalassemia Type", [0, 1, 2, 3],
                                   help="0 = normal, 1 = fixed defect, 2 = normal, "
                                        "3 = reversible defect")

    st.markdown("")
    _, mid, _ = st.columns([2, 1, 2])
    assess_btn = mid.button("🩺  Assess Health Risk", key="assess_btn")

    # ---------- Prediction ----------
    if assess_btn and MODEL_READY:
        input_df = pd.DataFrame([row])[FEATURE_ORDER]
        scaled = scaler.transform(input_df.values)

        progress_text = "Assessing patient's heart condition…"
        bar = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.01)
            bar.progress(i + 1, text=progress_text)
        time.sleep(0.2)
        bar.empty()

        try:
            prediction = int(model.predict(scaled)[0])
            try:
                proba = float(model.predict_proba(scaled)[0][1])
            except Exception:
                proba = float(prediction)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        pct = proba * 100
        is_positive = prediction == 1

        st.markdown("---")
        st.markdown("### 📋 Prediction Result")

        g1, g2 = st.columns([1.15, 1])

        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 52, "color": "#e6edf7"}},
                title={"text": "Predicted probability of heart disease",
                       "font": {"size": 14, "color": "#93a4bd"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": "#ef4444" if is_positive else "#22c55e",
                            "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],   "color": "rgba(34,197,94,.20)"},
                        {"range": [35, 65],  "color": "rgba(234,179,8,.20)"},
                        {"range": [65, 100], "color": "rgba(239,68,68,.22)"},
                    ],
                    "threshold": {"line": {"color": "#f8fafc", "width": 4},
                                  "thickness": 0.85, "value": 50},
                },
            ))
            gauge.update_layout(
                height=340, margin=dict(t=60, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7"},
            )
            st.plotly_chart(gauge, use_container_width=True)

        with g2:
            if is_positive:
                color = "#ef4444"
                label = "HEART DISEASE DETECTED"
                note = "Model classifies this profile as positive."
            else:
                color = "#22c55e"
                label = "NO HEART DISEASE DETECTED"
                note = "Model classifies this profile as negative."

            st.markdown(
                f"""<div class="verdict" style="background:linear-gradient(135deg,{color}2e,{color}12);
                    border-color:{color}66">
                  <div class="kpi-label" style="color:{color}">Model verdict</div>
                  <h2 style="color:{color}">{label}</h2>
                  <p><b>P = {pct:.1f}%</b> · threshold at 50%</p>
                  <p>{note}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            odds = proba / max(1e-9, (1 - proba))
            k1, k2 = st.columns(2)
            k1.markdown(
                f"<div class='card'><div class='kpi-label'>Odds</div>"
                f"<div class='kpi-value'>{odds:.2f}:1</div>"
                f"<div class='kpi-sub'>in favour of disease</div></div>",
                unsafe_allow_html=True)
            k2.markdown(
                f"<div class='card'><div class='kpi-label'>Margin to cutoff</div>"
                f"<div class='kpi-value'>{pct - 50:+.1f} pp</div>"
                f"<div class='kpi-sub'>percentage points</div></div>",
                unsafe_allow_html=True)

        # ---------- Patient summary ----------
        st.markdown("#### 📋 Patient Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Age", f"{row['age']} yrs")
        s2.metric("Sex", "Male" if row["sex"] == 1 else "Female")
        s3.metric("Resting BP", f"{row['trestbps']} mm Hg")
        s4.metric("Cholesterol", f"{row['chol']} mg/dl")

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Max HR", f"{row['thalach']} bpm")
        s6.metric("ST Depression", f"{row['oldpeak']}")
        s7.metric("Major Vessels", f"{row['ca']}")
        s8.metric("Thalassemia", f"Type {row['thal']}")

        # ---------- Risk drivers ----------
        st.markdown("#### ⚖️ Directional Risk Drivers")
        drivers = compute_risk_drivers(row)
        if drivers:
            cols = st.columns(min(len(drivers), 3))
            for i, d in enumerate(drivers):
                with cols[i % len(cols)]:
                    cls = "driver-up" if d["direction"] == "up" else "driver-down"
                    arrow = "▲" if d["direction"] == "up" else "▼"
                    st.markdown(
                        f"""<div class="{cls}">
                          <div class="driver-tag">{arrow} {d['direction'].upper()}</div>
                          <div class="driver-title">{d['factor']}</div>
                          <div class="driver-tag">{d['impact']} impact</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No dominant risk driver identified for this profile.")

        # ---------- Clinical recommendations ----------
        st.markdown("---")
        st.markdown("### 🏥 Clinical Recommendations")

        if is_positive:
            st.error("🔴 **Heart Disease Detected** — please consult a healthcare provider.")
            st.markdown("#### 📋 Suggested Next Steps")
            st.info("""
**🏥 Medical Attention**
- **Consult a Cardiologist:** Schedule a check-up to validate these results.
- **Monitor Vitals:** Keep a log of your blood pressure and heart rate.

**🥗 Dietary Changes**
- **Heart-Healthy Diet:** Focus on whole grains, fruits, vegetables, and lean proteins.
- **Limit Sodium & Fats:** Reduce intake of processed foods, salt, and saturated fats.

**🏃‍♂️ Lifestyle Adjustments**
- **Physical Activity:** 30 minutes of moderate exercise (walking, swimming) if approved by your doctor.
- **Stress Management:** Practice deep breathing or meditation to lower cortisol levels.
""")
        else:
            st.success("🟢 **No Heart Disease Detected** — keep up the healthy habits!")
            st.markdown("#### 📋 Maintenance Tips")
            st.info("""
**🏃‍♂️ Maintenance Routine**
- **Stay Active:** Continue your regimen of at least 30 minutes of moderate activity daily.
- **Regular Checkups:** Even with good results, schedule annual physicals to track trends.

**🥗 Nutrition & Immunity**
- **Balanced Plate:** Keep prioritising fruits, vegetables, and whole grains.
- **Hydration:** Drink plenty of water and limit sugary beverages.

**🧘 Mental Wellbeing**
- **Stress Management:** High stress can impact heart health over time — keep practising mindfulness.
- **Sleep:** Aim for 7–9 hours of quality sleep to help your body recover.
""")

        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer:** This tool is for educational and informational purposes only. "
            "A healthcare provider is required for a complete medical assessment. "
            "For a confirmed diagnosis and personalised treatment plan, please visit a medical professional."
        )

        report_df = pd.DataFrame([{
            **row,
            "predicted_class": "Positive" if is_positive else "Negative",
            "probability": round(proba, 6),
            "probability_pct": round(pct, 2),
        }])
        st.download_button(
            "⬇️ Download this assessment (CSV)",
            report_df.to_csv(index=False),
            file_name="riskpulse_assessment.csv",
            mime="text/csv",
        )

    elif assess_btn and not MODEL_READY:
        st.error(
            "Model artifacts are not loaded. Make sure `model.pkl` and `scaler.pkl` "
            "exist in the app directory."
        )

# =============================================================================
# 8.2  MODEL INSIGHTS
# =============================================================================
with tab_insights:
    st.markdown("### 📊 Model Performance & Insights")

    # ---------- Top-4 features strip ----------
    st.markdown("##### 🏆 Top 4 Predictive Features")
    t1, t2, t3, t4 = st.columns(4)
    for col, name, val in [
        (t1, "Thalassemia",     "17.0%"),
        (t2, "Chest Pain Type", "13.0%"),
        (t3, "ST Depression",   "12.1%"),
        (t4, "Max Heart Rate",  "10.9%"),
    ]:
        col.markdown(
            f"<div class='card' style='text-align:center'>"
            f"<div class='kpi-label'>{name}</div>"
            f"<div class='kpi-value'>{val}</div>"
            f"<div class='kpi-sub'>of model importance</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

    # ---------- Overall metrics ----------
    st.markdown("##### 📈 Overall Performance")
    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, sub in [
        (m1, "Accuracy",            MODEL_METRICS["accuracy"],  "Overall correctness"),
        (m2, "Sensitivity (Recall)", MODEL_METRICS["recall_1"],  "True positive rate"),
        (m3, "Specificity",          MODEL_METRICS["recall_0"],  "True negative rate"),
        (m4, "F1 (macro)",           MODEL_METRICS["f1_macro"], "Balanced F1"),
    ]:
        col.markdown(
            f"<div class='card'><div class='kpi-label'>{lbl}</div>"
            f"<div class='kpi-value'>{val}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Metrics evaluated on a held-out test set of {MODEL_METRICS['test_n']} samples. "
        "Because the holdout is small, treat these numbers as indicative rather than "
        "definitive — a larger cohort would tighten the confidence intervals considerably."
    )

    # ---------- Full classification report ----------
    st.markdown("---")
    st.markdown("##### 🧾 Full Classification Report")
    report_data = pd.DataFrame({
        "Class": ["0 · No Disease", "1 · Disease", "Macro avg", "Weighted avg"],
        "Precision": [0.87, 0.84, 0.85, 0.85],
        "Recall":    [0.84, 0.87, 0.85, 0.85],
        "F1-Score":  [0.85, 0.85, 0.85, 0.85],
        "Support":   [31, 30, 61, 61],
    })
    st.dataframe(report_data, use_container_width=True, hide_index=True)

    # ---------- Feature importance chart ----------
    st.markdown("---")
    st.markdown("#### 🌲 Feature Importance — What the Model Learned")
    imp_sorted = FEATURE_IMPORTANCE.sort_values("Importance")
    fig = px.bar(
        imp_sorted, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Teal",
    )
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    fig.update_layout(
        height=520, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.45)", font={"color": "#cbd5e1"},
        coloraxis_showscale=False, margin=dict(t=20, b=40, l=10, r=80),
        xaxis_title="Relative importance",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Importances are the Random Forest's Gini-based feature importances, summed "
        "across all trees. The top four features — Thalassemia, Chest Pain Type, ST "
        "Depression, and Max Heart Rate — account for over half of the model's total "
        "discriminative power."
    )

    # ---------- What each feature means ----------
    st.markdown("---")
    st.markdown("#### 🧭 What Each Feature Means")
    st.markdown("""
- **Thalassemia (thal)** — Nuclear stress test result. A reversible defect indicates ischemia; the single strongest predictor in this model.
- **Chest Pain Type (cp)** — Categorised as typical angina, atypical angina, non-anginal, or asymptomatic. Asymptomatic cases often indicate silent ischemia.
- **ST Depression (oldpeak)** — ST-segment depression induced by exercise, measured in millimetres. Higher values = more concerning.
- **Max Heart Rate (thalach)** — Peak heart rate during stress testing. Higher values are generally protective.
- **Major Vessels (ca)** — Number of major coronary vessels coloured by fluoroscopy. More vessels = higher risk.
- **Exercise Angina (exang)** — Chest pain provoked by exercise; a strong predictor of coronary disease.
- **Slope of ST Segment** — Shape of the ST segment during peak exercise.
- **Age & Sex** — Baseline demographic risk factors.
- **Serum Cholesterol & Resting BP** — Classic cardiovascular risk markers.
- **Resting ECG** — Resting electrocardiogram result.
- **Fasting Blood Sugar** — Indicator of diabetes; smallest contributor in this model.
    """)

# =============================================================================
# 8.3  ABOUT
# =============================================================================
with tab_about:
    a1, a2 = st.columns([1.5, 1])

    with a1:
        st.markdown("""
#### The Project
**RiskPulse** is a heart disease risk prediction tool built as a hands-on
machine learning deployment project. It uses a pre-trained classifier trained
on the **UCI Heart Disease dataset** — a standard benchmark in medical ML —
to estimate the probability that a patient has heart disease based on
13 routine clinical parameters.

#### Why It Matters
Cardiovascular disease remains the leading cause of death worldwide. Early
risk identification allows for timely intervention — lifestyle changes,
medication, or further diagnostic testing — that can meaningfully change
outcomes. Machine learning models like this one can serve as **decision
support**, flagging high-risk patients for clinician review without replacing
the clinician's judgement.

#### The Pipeline
1. **Data:** UCI Heart Disease dataset (Cleveland subset).
2. **Preprocessing:** StandardScaler for continuous features; categorical
   features encoded as integers.
3. **Model:** Random Forest Classifier (ensemble of decision trees).
4. **Evaluation:** Accuracy, Recall, Precision, F1-score on a held-out test set.
5. **Deployment:** Streamlit + pickle for model serialisation.

#### Limitations
- Trained on a relatively small historical dataset (303 patients).
- Not validated on Nigerian or African populations.
- Should be used for **education and demonstration only**.
        """)
        st.info(
            "This tool does **not** replace clinical judgement, ECG analysis, "
            "echocardiography, or a full cardiology workup.",
            icon="ℹ️",
        )

    with a2:
        st.markdown("""
<div class="card">
  <div class="kpi-label">Author</div>
  <div class="kpi-value" style="font-size:1.35rem">Samuel Ajeboriogbon</div>
  <div class="kpi-sub">pka <b>SBM ~ Nucopia</b></div>
  <hr>
  <div class="small">
  🎓 B.Sc. Statistics<br>
  🏛️ Obafemi Awolowo University, Ile-Ife<br>
  💼 Data Scientist &amp; Machine Learning Engineer<br>
  🚀 Inspired by TechCrush
  </div>
  <hr>
  <div class="kpi-label">Stack</div>
  <div class="small">Python · pandas · scikit-learn · Plotly · Streamlit ·
  Hugging Face Spaces</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="card" style="margin-top:14px">
  <div class="kpi-label">Ethics & Disclaimer</div>
  <div class="small" style="line-height:1.6">
  This is an <b>educational tool</b>. Predictions are not medical advice.
  Always consult a qualified healthcare provider for diagnosis and treatment.
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown(
    "<hr>"
    "<div style='text-align:center' class='small'>"
    "RiskPulse · Heart Disease Risk Intelligence · "
    "Built by <b>Samuel Ajeboriogbon (SBM ~ Nucopia)</b> · "
    "Obafemi Awolowo University, Ile-Ife"
    "</div>",
    unsafe_allow_html=True,
)
