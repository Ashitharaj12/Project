import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.title("Air Quality Prediction in Urban Areas")

# Load dataset
uploaded_file = st.file_uploader("Upload air quality CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Show first few rows
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    # Prepare data for training
    features = ["Traffic_Volume", "Temperature", "Humidity", "Wind_Speed", "NO2", "CO", "PM10"]
    target = "PM2.5"
    X = data[features]
    y = data[target]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, "air_quality_model.pkl")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**R-squared Score:** {r2:.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    ax.set_title("Feature Importance in Air Quality Prediction")
    st.pyplot(fig)
    
    # User inputs
    st.subheader("Enter Input Data")
    traffic_volume = st.number_input("Traffic Volume", min_value=100, max_value=5000, value=1000)
    temperature = st.number_input("Temperature (°C)", min_value=5.0, max_value=45.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=5.0)
    no2 = st.number_input("NO2 Level", min_value=0.0, max_value=500.0, value=50.0)
    co = st.number_input("CO Level", min_value=0.0, max_value=10.0, value=1.0)
    pm10 = st.number_input("PM10 Level", min_value=0.0, max_value=500.0, value=80.0)
    
    # Load trained model
    model = joblib.load("air_quality_model.pkl")
    
    # Prediction button
    if st.button("Predict PM2.5 Levels"):
        input_data = np.array([[traffic_volume, temperature, humidity, wind_speed, no2, co, pm10]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.write(f"Predicted PM2.5 Level: {prediction[0]:.2f} µg/m³")
else:
    st.write("Please upload a CSV file to proceed.")
