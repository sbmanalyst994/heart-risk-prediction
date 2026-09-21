# 🫀 RiskPulse — Heart Disease Risk Predictor

**An explainable machine-learning dashboard for predicting heart disease risk.**

Built by **[Ajeboriogbon Samuel A.](https://samuel-a.vercel.app)** (SBM ~ Nucopia) — B.Sc. Statistics, Obafemi Awolowo University (OAU), Ile-Ife, Nigeria.

🔗 **Live App:** [riskpulse.streamlit.app](https://riskpulse.streamlit.app)

---

## 📖 Overview
RiskPulse is a Streamlit-based clinical decision-support tool that predicts the likelihood of heart disease using 13 standard clinical parameters. It uses a **Random Forest Classifier** trained on the UCI Heart Disease dataset to provide instant, interpretable risk assessments.

## ✨ Key Features
- **Probability Gauge:** Instant visual risk score (0–100%).
- **Risk Drivers:** Explains which inputs pushed the risk up or down.
- **Patient Summary:** Quick reference card of the patient's profile.
- **Model Insights:** Built-in metrics and feature importance charts.
- **Export:** Download individual assessments as CSV.

## 📊 Model Performance
Evaluated on a held-out test set of 61 samples:
- **Accuracy:** 85%
- **Sensitivity (Recall):** 0.87
- **Specificity:** 0.84
- **F1-Score (Macro):** 0.85

**Top 4 Predictive Features:**
1. Thalassemia (17.0%)
2. Chest Pain Type (13.0%)
3. ST Depression (12.1%)
4. Max Heart Rate (10.9%)

## 🚀 Quick Start
To run this app locally, clone the repository, install dependencies, then launch Streamlit:

```bash
git clone https://github.com/sbmanalyst994/heart-risk-prediction.git
cd heart-risk-prediction
pip install -r requirements.txt
streamlit run app.py
```

**Required Files:**
- `app.py` — Streamlit dashboard
- `model.pkl` — Trained Random Forest classifier
- `scaler.pkl` — Fitted StandardScaler
- `requirements.txt` — Python dependencies
- `Heart Disease.ipynb` — Training & evaluation notebook

---

## 👨‍💻 About the Author
**Ajeboriogbon Samuel A.** (pka **SBM ~ Nucopia**)
- 🎓 B.Sc. Statistics, Obafemi Awolowo University (OAU), Ile-Ife
- 💼 Data Scientist & Machine Learning Engineer
- 🚀 Inspired by TechCrush

**Keywords:** Ajeboriogbon Samuel, SBM Nucopia, OAU Ile-Ife, Nigeria data scientist, machine learning engineer Nigeria, heart disease prediction, Streamlit dashboard, Random Forest, healthcare AI Africa.

## ⚠️ Disclaimer
This tool is for **educational purposes only**. It is not a medical device and should never replace professional medical diagnosis. Always consult a qualified healthcare provider.
