import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("House Price Prediction App")

# Create input fields
st.header("Enter House Features")

input_data = {}
for col in columns:
    if 'age_binned' in col or 'waterfront' in col or col in ['view', 'grade', 'condition']:
        input_data[col] = st.selectbox(f"{col}", options=[0, 1])
    else:
        input_data[col] = st.number_input(f"{col}", min_value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])
# Ensure missing columns (from one-hot) are filled with 0
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
