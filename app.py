import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up Streamlit page config
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("House Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/asha-hiremath04/House-Price-Prediction-Project/refs/heads/main/kc_house_data.csv'
    df = pd.read_csv(url)
    df.drop_duplicates(inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.year

    if 'yr_built' in df.columns:
        df['age_binned'] = pd.cut(df['yr_built'], bins=[df['yr_built'].min(), 1950, 1980, df['yr_built'].max()],
                                  labels=['Old', 'Mid-aged', 'New'], include_lowest=True)
        df = pd.get_dummies(df, columns=['age_binned'], drop_first=True)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['sqft_living', 'bedrooms']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['sqft_living', 'bedrooms']))
    df = pd.concat([df, poly_df[['sqft_living^2', 'bedrooms^2', 'sqft_living bedrooms']]], axis=1)

    return df

df = load_data()

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Select a Model",
    ("Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "XGBoost")
)

# Sidebar for input
st.sidebar.header("Enter Features")
def user_input_features():
    bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)
    sqft_living = st.sidebar.slider("Sqft Living", 500, 10000, 2000)
    grade = st.sidebar.slider("Grade", 1, 13, 7)
    view = st.sidebar.slider("View", 0, 4, 0)
    condition = st.sidebar.slider("Condition", 1, 5, 3)
    yr_built = st.sidebar.slider("Year Built", 1900, 2015, 2000)
    waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
    age_binned_mid = 1 if 1950 < yr_built <= 1980 else 0
    age_binned_new = 1 if yr_built > 1980 else 0

    sqft_living_sq = sqft_living ** 2
    bedrooms_sq = bedrooms ** 2
    sqft_living_bedrooms = sqft_living * bedrooms

    data = {
        "bedrooms": bedrooms,
        "sqft_living": sqft_living,
        "grade": grade,
        "view": view,
        "condition": condition,
        "yr_built": yr_built,
        "waterfront": waterfront,
        "sqft_living^2": sqft_living_sq,
        "bedrooms^2": bedrooms_sq,
        "sqft_living bedrooms": sqft_living_bedrooms,
        "age_binned_Mid-aged": age_binned_mid,
        "age_binned_New": age_binned_new
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prepare training data
X = df.drop(columns=["price", "id", "lat", "long", "zipcode"], errors="ignore")
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit selected model
model_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}

model = model_dict[model_option]
model.fit(X_train_scaled, y_train)

# Scale input and make prediction
input_scaled = scaler.transform(input_df[X.columns])
prediction = model.predict(input_scaled)

st.subheader("Predicted House Price")
st.write(f"${prediction[0]:,.2f}")

# Optionally show data and model performance
if st.checkbox("Show Training Data"):
    st.write(df.head())

if st.checkbox("Show Model Performance"):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    st.write("R² Score:", round(r2_score(y_test, preds), 4))
    st.write("MAE:", round(mean_absolute_error(y_test, preds), 2))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 2))