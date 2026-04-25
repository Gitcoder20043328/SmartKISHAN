import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Smart Crop Recommender", page_icon="🌾")

# 1. Load the 'Brain'
@st.cache_resource
def load_assets():
    return joblib.load('crop_model_final.pkl')

assets = load_assets()

st.title("🌾 Smart Crop Recommendation System")
st.markdown("Enter your farm details below to get the best crop suggestions.")

# 2. Sidebar / Inputs
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state = st.selectbox("State", assets['le_dict']['state'].classes_)
        season = st.selectbox("Season", assets['le_dict']['season'].classes_)
        soil_type = st.selectbox("Soil Type", assets['le_dict']['soiltype'].classes_)
        croptype = st.selectbox("Crop Category", assets['le_dict']['croptype'].classes_)

    with col2:
        temp = st.number_input("Temperature (°C)", value=28.0)
        humidity = st.number_input("Humidity (%)", value=70.0)
        rainfall = st.number_input("Rainfall (mm)", value=120.0)
        ph = st.number_input("Soil pH", value=6.5)

    with col3:
        n = st.number_input("Nitrogen (N)", value=90)
        p = st.number_input("Phosphorus (P)", value=45)
        k = st.number_input("Potassium (K)", value=55)
        area = st.number_input("Area (Ha)", value=100.0)

    submitted = st.form_submit_button("Get Recommendations")

# 3. Prediction Logic
if submitted:
    # Prepare Input
    inp = {
        'temperature(c)': temp, 'tempanomaly(c)': 0.0, 'rainfall(mm)': rainfall,
        'humidity(%)': humidity, 'soilph': ph, 'soilmoisture': 0.4,
        'soiltype': assets['le_dict']['soiltype'].transform([soil_type]),
        'n': n, 'p': p, 'k': k, 'fertilizerconsumption(kg/ha)': 100.0,
        'month': 6, 'season': assets['le_dict']['season'].transform([season]),
        'state': assets['le_dict']['state'].transform([state]),
        'croptype': assets['le_dict']['croptype'].transform([croptype]),
        'yield': 2.5, 'area': area,
    }
    
    # Scale and Predict
    row_df = pd.DataFrame([inp])[assets['features']]
    row_scaled = assets['scaler'].transform(row_df)
    probs = assets['model'].predict_proba(row_scaled)[0]
    
    # Show Results
    top3_idx = np.argsort(probs)[::-1][:3]
    
    st.subheader("Results")
    for i, idx in enumerate(top3_idx):
        crop = assets['le_crop'].classes_[idx]
        conf = probs[idx] * 100
        if conf > 0.1:
            st.success(f"**#{i+1}: {crop}** ({conf:.1f}% Confidence)")
            st.info(f"📋 **Advisory:** {assets['advisory'].get(crop, 'Standard practices apply.')}")
import os

@st.cache_resource
def load_assets():
    # This creates the correct path to the folder: exported_models/crop_model_final.pkl
    # Using os.path.join makes it work on both Windows and Vercel/Linux
    file_path = os.path.join('exported_models', 'crop_model_final.pkl')
    
    if not os.path.exists(file_path):
        st.error(f"❌ File not found at {file_path}. Please check your folder structure!")
        st.stop()
        
    return joblib.load(file_path)
