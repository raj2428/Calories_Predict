# calories_app.py
import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load trained model
model = pickle.load(open('calories_model.pkl', 'rb'))

# App Title
st.title("🔥 Calories Burn Predictor")
st.write("Enter your details to predict calories burned during exercise.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=170.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=10.0,
    max_value=300.0,
    value=70.0
)

duration = st.number_input(
    "Exercise Duration (minutes)",
    min_value=1,
    max_value=500,
    value=30
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=40,
    max_value=220,
    value=90
)

body_temp = st.number_input(
    "Body Temperature (°C)",
    min_value=35.0,
    max_value=45.0,
    value=37.0
)

# Prediction Button
if st.button("Predict Calories Burned"):

    # Encode Gender
    gender_encoded = 1 if gender == "Male" else 0

    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)

    # Create DataFrame with EXACT feature names
    features = pd.DataFrame([{
        'Gender': gender_encoded,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'BMI': bmi
    }])

    # Prediction
    prediction = model.predict(features)

    # Output
    st.success(f"🔥 Estimated Calories Burned: {prediction[0]:.2f} kcal")

    # BMI Display
    st.info(f"📊 Calculated BMI: {bmi:.2f}")

# Footer
st.write("---")
st.caption(f"Scikit-learn Version: {sklearn.__version__}")