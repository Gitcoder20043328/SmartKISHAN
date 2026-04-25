```python
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="SmartKISHAN", page_icon="🌾", layout="centered")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_assets():
    return joblib.load("crop_model_final.pkl")

assets = load_assets()

# -------------------------------
# Title
# -------------------------------
st.title("🌾 SmartKISHAN")
st.write("AI Based Crop Recommendation System")

# -------------------------------
# Input Form
# -------------------------------
with st.form("my_form"):

    state = st.selectbox(
        "Select State",
        assets['le_dict']['state'].classes_
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    nitrogen = st.number_input("Nitrogen", value=50)
    phosphorus = st.number_input("Phosphorus", value=40)
    potassium = st.number_input("Potassium", value=40)

    submitted = st.form_submit_button("Get Recommendation")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    try:
        # Encode State
        state_encoded = assets['le_dict']['state'].transform([state])[0]

        # Input Dictionary
        inp = {
            "state": state_encoded,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium
        }

        # DataFrame in Correct Feature Order
        row_df = pd.DataFrame([inp])[assets['features']]

        # Scale Input
        row_scaled = assets['scaler'].transform(row_df)

        # Predict Probability
        probs = assets['model'].predict_proba(row_scaled)[0]

        # Top 3 Predictions
        top3_idx = np.argsort(probs)[::-1][:3]

        st.success("Top Crop Recommendations")

        for i in top3_idx:
            crop = assets['model'].classes_[i]
            confidence = probs[i] * 100
            st.write(f"🌱 {crop} : {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
```
