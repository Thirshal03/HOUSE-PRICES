import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('house_price_model.pkl')

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")
st.write("Enter the details of the house:")

# Input fields for features
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
sqft_living = st.slider("Living Area (sqft)", 500, 10000, 2000)
floors = st.selectbox("Number of Floors", [1, 2, 3])
waterfront = st.radio("Waterfront View", [0, 1], format_func=lambda x: "Yes" if x else "No")
view = st.slider("View Rating", 0, 4, 1)
grade = st.slider("Construction Grade (1-13)", 1, 13, 7)
yr_built = st.slider("Year Built", 1900, 2023, 2000)

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[bedrooms, bathrooms, sqft_living, floors, waterfront, view, grade, yr_built]],
                            columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'grade', 'yr_built'])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
