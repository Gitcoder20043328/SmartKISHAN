import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="KrishiDarshan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Remove default streamlit padding */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Header ── */
.header-wrap {
    background: linear-gradient(135deg, #1a3c2e 0%, #2d6a4f 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.header-icon { font-size: 3rem; }
.header-title { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.header-sub   { color: #95d5b2; font-size: 0.9rem; margin: 0.3rem 0 0; }
.header-badge {
    background: rgba(255,255,255,0.15);
    color: #d8f3dc;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.5rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #52b788;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e8f5e9;
}

/* ── Form card ── */
.form-card {
    background: #fff;
    border: 1px solid #e8f5e9;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Submit button override ── */
div[data-testid="stFormSubmitButton"] button {
    background: #2d6a4f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: background 0.2s;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1b4332 !important;
}

/* ── Result cards ── */
.result-rank-1 { border-left: 5px solid #2d6a4f; background: #f0faf4; }
.result-rank-2 { border-left: 5px solid #52b788; background: #f6fcf8; }
.result-rank-3 { border-left: 5px solid #95d5b2; background: #fafffe; }
.result-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-crop   { font-size: 1.2rem; font-weight: 700; color: #1b4332; }
.result-conf   { font-size: 0.85rem; color: #555; margin-top: 0.2rem; }
.conf-bar-wrap { background: #e8f5e9; border-radius: 20px; height: 6px; margin: 0.6rem 0; overflow: hidden; }
.conf-bar      { height: 6px; border-radius: 20px; background: linear-gradient(90deg, #52b788, #2d6a4f); }
.advisory-box  {
    background: #fffef5;
    border: 1px solid #f0e68c;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #555;
    margin-top: 0.8rem;
    line-height: 1.6;
}

/* ── Stat pills ── */
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-pill {
    background: #f0faf4;
    border: 1px solid #b7e4c7;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: #2d6a4f;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    return joblib.load('crop_model_final.pkl')

assets = load_assets()


# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="header-icon">🌾</div>
  <div>
    <div class="header-title">KrishiDarshan</div>
    <div class="header-sub">Climate-Aware Crop Recommendation · Manav Rachna University DTI Project 2024</div>
    <span class="header-badge">35,364 records · 47 crops · 6 states · 2011–2025</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Input Form ─────────────────────────────────────────────────
with st.form("crop_form"):
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="section-label">Location & Season</div>', unsafe_allow_html=True)
        state    = st.selectbox("State",         assets['le_dict']['state'].classes_,   label_visibility="visible")
        season   = st.selectbox("Season",        assets['le_dict']['season'].classes_,  label_visibility="visible")
        month    = st.selectbox("Month", list(range(1, 13)),
                   format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                           "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        soiltype = st.selectbox("Soil Type",     assets['le_dict']['soiltype'].classes_)
        croptype = st.selectbox("Crop Category", assets['le_dict']['croptype'].classes_)

    with col2:
        st.markdown('<div class="section-label">Climate Conditions</div>', unsafe_allow_html=True)
        temp     = st.slider("Temperature (°C)",    min_value=5.0,  max_value=50.0, value=28.0, step=0.5)
        rainfall = st.slider("Rainfall (mm)",        min_value=0.0,  max_value=500.0,value=120.0,step=5.0)
        humidity = st.slider("Humidity (%)",         min_value=10.0, max_value=100.0,value=70.0, step=1.0)
        ph       = st.slider("Soil pH",              min_value=4.0,  max_value=9.5,  value=6.5,  step=0.1)
        moisture = st.slider("Soil Moisture (0–1)",  min_value=0.0,  max_value=1.0,  value=0.4,  step=0.05)

    with col3:
        st.markdown('<div class="section-label">Soil Nutrients & Farm</div>', unsafe_allow_html=True)
        n          = st.number_input("Nitrogen — N (kg/ha)",    min_value=0,   max_value=200, value=90)
        p          = st.number_input("Phosphorus — P (kg/ha)",  min_value=0,   max_value=200, value=45)
        k          = st.number_input("Potassium — K (kg/ha)",   min_value=0,   max_value=200, value=55)
        fertilizer = st.number_input("Fertilizer (kg/ha)",      min_value=0.0, max_value=500.0,value=100.0)
        area       = st.number_input("Farm Area (Ha)",           min_value=1.0, max_value=10000.0, value=100.0)
        yield_val  = st.number_input("Expected Yield (T/Ha)",   min_value=0.1, max_value=30.0, value=2.5)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Analyse & Recommend Crop", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────
if submitted:
    with st.spinner("Running model..."):
        inp = {
            'temperature(c)':              float(temp),
            'tempanomaly(c)':              0.0,
            'rainfall(mm)':                float(rainfall),
            'humidity(%)':                 float(humidity),
            'soilph':                      float(ph),
            'soilmoisture':                float(moisture),
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

    top3_idx   = np.argsort(probs)[::-1][:3]
    rank_class = ["result-rank-1", "result-rank-2", "result-rank-3"]
    rank_label = ["Best Match", "2nd Choice", "3rd Choice"]

    st.markdown("### Recommended Crops")

    for rank, idx in enumerate(top3_idx):
        crop = assets['le_crop'].classes_[idx]
        conf = probs[idx] * 100
        adv  = assets['advisory'].get(crop, "Follow recommended practices for this crop.")
        bar  = min(int(conf), 100)

        st.markdown(f"""
        <div class="result-card {rank_class[rank]}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="result-crop">{crop}</div>
                    <div class="result-conf">{rank_label[rank]} &nbsp;·&nbsp; {conf:.1f}% confidence</div>
                </div>
                <div style="font-size:1.6rem;">{'🥇' if rank==0 else '🥈' if rank==1 else '🥉'}</div>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{bar}%;"></div>
            </div>
            <div class="advisory-box">
                <b>Farming Advisory:</b><br>{adv[:350]}{"..." if len(adv)>350 else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Input recap as clean pills
    st.markdown("<br><b>Your input snapshot</b>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-pill">📍 {state}</span>
        <span class="stat-pill">🗓 {season} · {["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1]}</span>
        <span class="stat-pill">🌡 {temp}°C</span>
        <span class="stat-pill">🌧 {rainfall} mm</span>
        <span class="stat-pill">💧 {humidity}%</span>
        <span class="stat-pill">🧪 pH {ph}</span>
        <span class="stat-pill">🌿 N{n} / P{p} / K{k}</span>
        <span class="stat-pill">🪨 {soiltype} soil</span>
        <span class="stat-pill">🌾 {croptype}</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#f8fffe; border:1px dashed #b7e4c7; border-radius:12px;
                padding:2rem; text-align:center; color:#555; margin-top:1rem;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🌱</div>
        <div style="font-weight:600; color:#2d6a4f; font-size:1.05rem;">
            Enter your field conditions above
        </div>
        <div style="font-size:0.88rem; margin-top:0.4rem; color:#888;">
            Fill in location, climate, and soil data — then click Analyse & Recommend Crop
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    KrishiDarshan <br>
    Kushal · Jayesh Sharma · Anirudh Singh · Remant Jha
</div>
""", unsafe_allow_html=True)
