import streamlit as st
# Set page config at the very top
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

import pickle
import numpy as np
import base64

# ------------------------ Background Image Function ------------------------
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background image (make sure image.png is in the same directory)
set_background("image.png")

# ------------------------ Load Model and Scaler ------------------------
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ------------------------ Prediction Function ------------------------
def predict(male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):
    
    features = np.array([[male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
                          prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)
    return result[0]

# ------------------------ Streamlit UI ------------------------
st.title("💓 Heart Disease Prediction App")
st.markdown("Fill out the details below to predict the risk of heart disease:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        male = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120)
        currentSmoker = st.selectbox("Current Smoker", options=["Yes", "No"])
        cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0.0)
        BPMeds = st.selectbox("BP Medications", options=["Yes", "No"])
        prevalentStroke = st.selectbox("Prevalent Stroke", options=["Yes", "No"])
        prevalentHyp = st.selectbox("Prevalent Hypertension", options=["Yes", "No"])
    
    with col2:
        diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
        totChol = st.number_input("Total Cholesterol", min_value=0.0)
        sysBP = st.number_input("Systolic BP", min_value=0.0)
        diaBP = st.number_input("Diastolic BP", min_value=0.0)
        BMI = st.number_input("BMI", min_value=0.0, step=0.01)
        heartRate = st.number_input("Heart Rate", min_value=0.0)
        glucose = st.number_input("Glucose Level", min_value=0.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Encode categorical inputs
        male_val = 1 if male == "Male" else 0
        currentSmoker_val = 1 if currentSmoker == "Yes" else 0
        BPMeds_val = 1 if BPMeds == "Yes" else 0
        prevalentStroke_val = 1 if prevalentStroke == "Yes" else 0
        prevalentHyp_val = 1 if prevalentHyp == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0

        result = predict(male_val, age, currentSmoker_val, cigsPerDay, BPMeds_val, prevalentStroke_val,
                         prevalentHyp_val, diabetes_val, totChol, sysBP, diaBP, BMI, heartRate, glucose)
        
        st.success("✅ The Patient has **Heart Disease**." if result == 1 else "✅ The Patient has **No Heart Disease**.")
