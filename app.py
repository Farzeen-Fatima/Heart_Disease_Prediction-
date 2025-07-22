import streamlit as st
import pandas as pd
import joblib
import os

# Load models and scaler safely
try:
    log_model = joblib.load("logistic_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Set these based on your training results ---
log_acc = 0.85   # replace with Logistic Regression test accuracy
svm_acc = 0.88   # replace with SVM test accuracy

st.title("Heart Disease Prediction App ❤️")

st.write("### Enter Patient Information:")

# Input fields (labels unchanged)
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)  # was trtbps
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)  # was thalachh
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])  # was exng
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 7.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])  # was slp
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", [0, 1, 2, 3, 4])  # was caa
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])  # was thall

# Prepare input data with correct column names
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                   "restecg", "thalach", "exang", "oldpeak",
                                   "slope", "ca", "thal"])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict with both models
if st.button("Predict with Both Models"):
    pred_log = log_model.predict(input_scaled)[0]
    pred_svm = svm_model.predict(input_scaled)[0]

    st.subheader("Predictions")
    st.write(f"**Logistic Regression:** {'Heart Disease' if pred_log == 1 else 'No Heart Disease'}")
    st.write(f"**SVM:** {'Heart Disease' if pred_svm == 1 else 'No Heart Disease'}")

    # Comparison table of accuracies
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "SVM"],
        "Test Accuracy": [log_acc, svm_acc]
    })
    st.table(comparison)

    # Decide which model is overall better
    better_model = "SVM" if svm_acc > log_acc else "Logistic Regression"
    st.info(f"Based on test accuracy, **{better_model}** performed better overall.")

    if pred_log == pred_svm:
        st.success(f"Both models agree: {'Heart Disease' if pred_log == 1 else 'No Heart Disease'}")
    else:
        st.warning("Models disagree! Consider the better-performing model's prediction.")
