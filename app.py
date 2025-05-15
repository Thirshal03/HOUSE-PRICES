import streamlit as st
import gdown
import joblib
import os
import pandas as pd

# Google Drive file IDs
model_id = "1H9BvURY5XejXmtSYcRyeahgGShWJn_Pu"
scaler_id = "1H3ncHCDTOWMuoMnoJUL3d0roKdMxCRtm"
columns_id = "1H9QE__qgGnoabLjFc3q1JB0BX5lP_Th5"

# Function to download files from Google Drive if not already present
def download_if_needed(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Download the necessary files
download_if_needed(model_id, "model.pkl")
download_if_needed(scaler_id, "scaler.pkl")
download_if_needed(columns_id, "columns.pkl")

# Load model, scaler, and columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Streamlit UI
st.title("House Price Prediction App")

st.write("Enter the feature values below:")

# Create input fields for all features
user_input = {}
for col in columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
