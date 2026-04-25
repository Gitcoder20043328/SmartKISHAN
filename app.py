import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Recommender",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
  .main-title { font-size:2.4rem; font-weight:bold; color:#1B4332; }
  .sub        { font-size:1rem; color:#555; margin-bottom:1rem; }
  .card       { background:#D8F3DC; padding:1.2rem; border-radius:12px;
                border-left:5px solid #1B4332; margin:0.6rem 0; }
  .advisory   { background:#FFFDE7; padding:0.8rem; border-radius:8px;
                font-size:0.88rem; color:#444; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌾 Smart Crop Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Climate Analysis for Intelligent Crop Recommendation in India · DTI Project 2024</div>", unsafe_allow_html=True)
st.markdown("---")


# ── Load model (only once) ────────────────────────────────────
@st.cache_resource
def load_assets():
    return joblib.load('crop_model_final.pkl')

assets = load_assets()


# ── Inputs ────────────────────────────────────────────────────
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location & Season")
        state    = st.selectbox("State",         assets['le_dict']['state'].classes_)
        season   = st.selectbox("Season",        assets['le_dict']['season'].classes_)
        soiltype = st.selectbox("Soil Type",     assets['le_dict']['soiltype'].classes_)
        croptype = st.selectbox("Crop Category", assets['le_dict']['croptype'].classes_)
        month    = st.selectbox("Month", list(range(1, 13)),
                   format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                           "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

    with col2:
        st.subheader("🌦️ Climate & Soil")
        temp          = st.number_input("Temperature (°C)",    value=28.0, min_value=5.0,  max_value=50.0)
        humidity      = st.number_input("Humidity (%)",        value=70.0, min_value=10.0, max_value=100.0)
        rainfall      = st.number_input("Rainfall (mm)",       value=120.0,min_value=0.0,  max_value=500.0)
        ph            = st.number_input("Soil pH",             value=6.5,  min_value=4.0,  max_value=9.5)
        soil_moisture = st.number_input("Soil Moisture (0-1)", value=0.4,  min_value=0.0,  max_value=1.0)

    with col3:
        st.subheader("🌿 Nutrients & Farm")
        n          = st.number_input("Nitrogen (N) kg/ha",    value=90,   min_value=0,   max_value=200)
        p          = st.number_input("Phosphorus (P) kg/ha",  value=45,   min_value=0,   max_value=200)
        k          = st.number_input("Potassium (K) kg/ha",   value=55,   min_value=0,   max_value=200)
        fertilizer = st.number_input("Fertilizer (kg/ha)",    value=100.0,min_value=0.0, max_value=500.0)
        area       = st.number_input("Farm Area (Ha)",         value=100.0,min_value=1.0, max_value=10000.0)
        yield_val  = st.number_input("Expected Yield (T/Ha)", value=2.5,  min_value=0.1, max_value=30.0)

    submitted = st.form_submit_button("🌾 Get Recommendations", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────
if submitted:
    with st.spinner("Analysing field conditions..."):

        # FIX: [0] at end of every transform → scalar not array
        inp = {
            'temperature(c)':              float(temp),
            'tempanomaly(c)':              0.0,
            'rainfall(mm)':                float(rainfall),
            'humidity(%)':                 float(humidity),
            'soilph':                      float(ph),
            'soilmoisture':                float(soil_moisture),
            'soiltype':                    int(assets['le_dict']['soiltype'].transform([soiltype])[0]),
            'n':                           float(n),
            'p':                           float(p),
            'k':                           float(k),
            'fertilizerconsumption(kg/ha)':float(fertilizer),
            'month':                       int(month),
            'season':                      int(assets['le_dict']['season'].transform([season])[0]),
            'state':                       int(assets['le_dict']['state'].transform([state])[0]),
            'croptype':                    int(assets['le_dict']['croptype'].transform([croptype])[0]),
            'yield':                       float(yield_val),
            'area':                        float(area),
        }

        row_df     = pd.DataFrame([inp])[assets['features']].astype(float)
        row_scaled = assets['scaler'].transform(row_df)
        probs      = assets['model'].predict_proba(row_scaled)[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    emojis   = ["🥇", "🥈", "🥉"]

    st.markdown("## 🏆 Top 3 Recommended Crops")
    for rank, idx in enumerate(top3_idx):
        crop = assets['le_crop'].classes_[idx]
        conf = probs[idx] * 100
        if conf > 0.1:
            adv = assets['advisory'].get(crop, "Follow standard recommended practices for this crop.")
            st.markdown(f"""
            <div class='card'>
                <b>{emojis[rank]} #{rank+1}: {crop}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Confidence: <b>{conf:.1f}%</b>
                <div class='advisory'>
                    📋 <b>Advisory:</b> {adv[:300]}{"..." if len(adv) > 300 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.success("Recommendation complete!")

    with st.expander("📊 View Input Summary"):
        st.dataframe(pd.DataFrame([{
            "State": state, "Season": season, "Month": month,
            "Soil Type": soiltype, "Crop Type": croptype,
            "Temp": temp, "Rainfall": rainfall, "Humidity": humidity,
            "pH": ph, "Moisture": soil_moisture,
            "N": n, "P": p, "K": k,
        }]), use_container_width=True)

else:
    st.info("Fill in your field conditions above and click Get Recommendations")
