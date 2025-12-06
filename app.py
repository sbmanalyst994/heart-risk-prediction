import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# X = X.rename(columns={
#     'age': 'Age',
#     'sex': 'Sex',
#     'cp': 'Chest Pain Type',
#     'trestbps': 'Resting BP',
#     'chol': 'Serum Cholesterol',
#     'fbs': 'Fasting Blood Sugar',
#     'restecg': 'Resting ECG',
#     'thalach': 'Max Heart Rate',
#     'exang': 'Exercise Angina',
#     'oldpeak': 'ST Depression',
#     'slope': 'Slope ST Segment',
#     'ca': 'Major Vessels',
#     'thal': 'Thalassemia'
# })

@st.cache_resource #tells Streamlit to load the model once and resuse them

def load_asset():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_asset()

# App confgn
st.set_page_config(
    page_title="RiskPulse: Heart Disease Predictor App",
    page_icon="heart_logo.png",
    layout="centered"
)

def main():
    st.image("heart_logo.png", use_container_width=True, width="stretch",
             caption='Heart Disease Prediction')
    st.title("Heart Disease Risk Prediction App")

    st.markdown("""
                :blue-background[***Kindly provide the required medical parameters to initiate the risk assessment.***]""")

    st.subheader("Patient Information")
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 0 if sex == "Female" else 1

    st.subheader("Heart Health Indicators")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 90)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 60)
    oldpeak = st.slider("ST Depression (0.0–6.0)", 0.0, 6.0, 1.5)
   
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    fbs = 0 if fbs == "False" else 1

    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    exang = 0 if exang == "No" else 1


    cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
    selected_cp = st.selectbox("Chest Pain type", cp_options)
    cp = cp_options.index(selected_cp)

    restecg_options = ["Normal", "Abnormal", "Structural Issues"]
    selected_restecg = st.selectbox("Resting ECG Results", restecg_options)
    restecg = restecg_options.index(selected_restecg)

    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia type", [0, 1, 2, 3])

    # Input data
    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_data = pd.DataFrame([input_dict])

    col = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    input_data_array = input_data.values
    scaled_data = scaler.transform(input_data_array)


    if st.button("Assess Health Risk Prediciton"):
        progress_text = "Assessing Patient's Heart condition..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.015)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.3)
        my_bar.empty()

        try:
            prediction = model.predict(scaled_data)

            custom = """<hr style="border: none; height: 3px; background-color: #333;" />"""
            st.markdown(custom, unsafe_allow_html=True)
            st.subheader(":blue[Prediction Assessment Result]")
            if prediction == 1:
                st.error("🔴 **Heart Disease Detected**")
                st.warning("⚠️ Please consult a healthcare provider for a full diagnosis and treatment plan.")
                st.subheader("📋 Suggested Next Steps")
                st.info("""
                **🏥 Medical Attention**
                * **Consult a Cardiologist:** Schedule a check-up immediately to validate these results.
                * **Monitor Vitals:** Keep a log of your blood pressure and heart rate.

                **🥗 Dietary Changes**
                * **Heart-Healthy Diet:** Focus on whole grains, fruits, vegetables, and lean proteins.
                * **Limit Sodium & Fats:** Reduce intake of processed foods, salt, and saturated fats.

                **🏃‍♂️ Lifestyle Adjustments**
                * **Physical Activity:** Aim for 30 minutes of moderate exercise (walking, swimming) if approved by your doctor.
                * **Stress Management:** Practice deep breathing or meditation to lower cortisol levels.
                """)
                st.markdown('---')
                st.caption("⚠️ **Disclaimer:** *This tool is for Educational and Informational purposes only.*\n"
                        "\n*A healthcare provider is required for a complete medical assessment.* "
                        "*For a confirmed diagnosis and personalized treatment plan, please visit a medical professional.*")
            else:
                st.success("🟢 **No Heart Disease Detected**")
                st.info("💪 Great job! Keep maintaining your heart health with these tips:")
                st.subheader("📋 More Tips for you")
                st.info("""
                **🏃‍♂️ Maintenance Routine**
                * **Stay Active:** Continue your regimen of at least 30 minutes of moderate activity daily.
                * **Regular Checkups:** Even with good results, schedule annual physicals to track trends.

                **🥗 Nutrition & Immunity**
                * **Balanced Plate:** Keep prioritizing fruits, vegetables, and whole grains.
                * **Hydration:** Drink plenty of water and limit sugary beverages.

                **🧘 Mental Wellbeing**
                * **Stress Management:** High stress can impact heart health over time; keep practicing mindfulness.
                * **Sleep:** Aim for 7-9 hours of quality sleep to help your body recover.
                """)
                st.markdown('---')
                st.caption("⚠️ **Disclaimer:** *This tool is for Educational and Informational purposes only.*\n"
                        "\n*A healthcare provider is required for a complete medical assessment.* "
                        "*For a confirmed diagnosis and personalized treatment plan, please visit a medical professional.*")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


    with st.sidebar:
        st.markdown("---")
        st.header("👨‍💻 About the Developer")
        st.write("""
        **Samuel Ajeboriogbon** (*popularly known as SBM Nucopia*) is a Part 4 Statistics student at the prestigious 
        **Obafemi Awolowo University (OAU)**. 
        
        Currently studying in the Department of Mathematics, he applies critical thinking
        to solve complex problems and is actively pursuing a career in Data Science.
        """)

        st.markdown("---")
        st.header("👨‍💻 About the App")
        st.write("""
        This app utilises a pre-trained Machine Learning model, developed using Kaggle's medical datasets,
                 to predict the likelihood of heart disease.
        """)
        
        st.markdown("---")
        st.caption("*This project is proudly Inspired by TechCrush*")
        st.markdown('---')
        st.caption("⚠️ **Disclaimer:** *This tool is for Educational and Informational purposes only.*")
        st.caption("*A healthcare provider is required for a complete medical assessment.* "
                   "*For a confirmed diagnosis and personalized treatment plan,* " \
                   "*please visit a medical professional.*")


if __name__ == "__main__":
    main()