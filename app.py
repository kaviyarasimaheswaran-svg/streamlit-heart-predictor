import streamlit as st
import pandas as pd
import pickle

# Load the trained KNN model
with open("heart_knn_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1])  # 0 = female, 1 = male
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=["age","sex","cp","trestbps","chol","fbs",
                                       "restecg","thalach","exang","oldpeak","slope","ca","thal"])
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
