import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="CareVision - Health Risk Predictor", page_icon="🩺", layout="centered")

ART = os.path.join(os.path.dirname(__file__), 'artifacts')

@st.cache_resource
def load_artifacts():
    model = joblib.load(f'{ART}/carevision_model.pkl')
    scaler = joblib.load(f'{ART}/scaler.pkl')
    encoders = joblib.load(f'{ART}/encoders.pkl')
    target_le = joblib.load(f'{ART}/target_encoder.pkl')
    feature_cols = joblib.load(f'{ART}/feature_cols.pkl')
    best_model_name = joblib.load(f'{ART}/best_model_name.pkl')
    return model, scaler, encoders, target_le, feature_cols, best_model_name

model, scaler, encoders, target_le, feature_cols, best_model_name = load_artifacts()

st.title("🩺 CareVision")
st.subheader("Intelligent Health Monitoring System")
st.markdown("AI-powered early health risk prediction based on patient vitals, lab values, and lifestyle data.")
st.divider()

with st.form("patient_form"):
    st.markdown("### Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 110, 35)
        gender = st.selectbox("Gender", encoders['gender'].classes_)
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 220, 120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 75)
        glucose_mg_dl = st.number_input("Glucose (mg/dL)", 50, 400, 100)
        hba1c_pct = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5, step=0.1)
    with col2:
        cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 180)
        smoking_status = st.selectbox("Smoking Status", encoders['smoking_status'].classes_)
        exercise_days_wk = st.slider("Exercise Days/Week", 0, 7, 3)
        sleep_hrs_night = st.number_input("Sleep Hours/Night", 2.0, 12.0, 7.0, step=0.5)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        primary_symptom = st.selectbox("Primary Symptom", encoders['primary_symptom'].classes_)
        city = st.selectbox("City", encoders['city'].classes_)

    submitted = st.form_submit_button("🔍 Predict Health Risk", use_container_width=True)

if submitted:
    input_dict = {
        'age': age, 'gender': gender, 'bmi': bmi, 'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp, 'heart_rate': heart_rate, 'glucose_mg_dl': glucose_mg_dl,
        'hba1c_pct': hba1c_pct, 'cholesterol_total': cholesterol_total,
        'smoking_status': smoking_status, 'exercise_days_wk': exercise_days_wk,
        'sleep_hrs_night': sleep_hrs_night, 'stress_level': stress_level,
        'primary_symptom': primary_symptom, 'city': city
    }
    row = {}
    for col in feature_cols:
        val = input_dict[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        row[col] = val
    X_input = pd.DataFrame([row])[feature_cols]

    if best_model_name == 'Logistic Regression':
        X_input_proc = scaler.transform(X_input)
    else:
        X_input_proc = X_input

    pred = model.predict(X_input_proc)[0]
    pred_label = target_le.inverse_transform([pred])[0]
    proba = model.predict_proba(X_input_proc)[0]

    st.divider()
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'red'}
    st.markdown(f"## Predicted Risk Level: :{color_map.get(pred_label,'blue')}[{pred_label}]")

    proba_df = pd.DataFrame({'Risk Level': target_le.classes_, 'Probability': proba}).sort_values('Probability', ascending=False)
    st.bar_chart(proba_df.set_index('Risk Level'))

    st.markdown("### Recommendation")
    recs = {
        'Low': "Health indicators are within normal range. Maintain current lifestyle with regular checkups.",
        'Medium': "Some risk factors detected. Consider lifestyle adjustments and a follow-up consultation.",
        'High': "Multiple risk factors present. Medical consultation is recommended soon.",
        'Critical': "Significant risk detected. Immediate medical attention is advised."
    }
    st.info(recs.get(pred_label, "Consult a healthcare professional for further evaluation."))
    st.caption(f"Model used: {best_model_name} | This is a decision-support tool, not a medical diagnosis.")

st.divider()
st.caption("CareVision Project | AI-ML Based Health Prediction Platform")
