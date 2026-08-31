import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="KrishiDarshan | Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 2rem; }

.header-wrap {
    background: linear-gradient(135deg, #163a2b 0%, #2d6a4f 58%, #40916c 100%);
    border-radius: 20px; padding: 1.8rem 2.2rem; margin-bottom: 1.25rem;
    display:flex; align-items:center; gap:1.25rem;
    box-shadow:0 8px 28px rgba(45,106,79,.16);
}
.header-icon {
    width:64px; height:64px; display:grid; place-items:center;
    background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.16);
    border-radius:16px; font-size:2.25rem; flex:0 0 auto;
}
.header-title { color:#fff; font-size:clamp(1.55rem,3vw,2.15rem); font-weight:700; line-height:1.15; }
.header-sub { color:#d8f3dc; font-size:.9rem; margin-top:.35rem; }
.header-badge {
    background:rgba(255,255,255,.14); color:#e9f8ed; font-size:.72rem;
    padding:.32rem .75rem; border-radius:20px; display:inline-block; margin-top:.6rem;
}
.info-strip {
    background:#f4fbf7; border:1px solid #d8eee0; border-radius:12px;
    padding:.8rem 1rem; margin-bottom:1rem; color:#315b47; font-size:.84rem; line-height:1.55;
}
.section-label {
    font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#2d6a4f; margin:.2rem 0 .7rem; padding-bottom:.45rem; border-bottom:1px solid #e1efe6;
}
.form-card {
    background:#fff; border:1px solid #e2eee6; border-radius:14px;
    padding:1.05rem 1.15rem .65rem; min-height:100%;
    box-shadow:0 3px 14px rgba(20,60,40,.035);
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius:9px; }
div[data-testid="stFormSubmitButton"] button {
    min-height:48px; background:#2d6a4f !important; color:white !important;
    border:none !important; border-radius:11px !important; font-weight:700 !important;
    font-size:.98rem !important; box-shadow:0 5px 15px rgba(45,106,79,.18);
}
div[data-testid="stFormSubmitButton"] button:hover { background:#1b4332 !important; transform:translateY(-1px); }
.result-card {
    border-radius:14px; padding:1rem 1.2rem; margin-bottom:.8rem;
    border:1px solid #e5ece7; box-shadow:0 3px 14px rgba(20,60,40,.04);
}
.result-rank-1 { border-left:5px solid #2d6a4f; background:#f0faf4; }
.result-rank-2 { border-left:5px solid #52b788; background:#f6fcf8; }
.result-rank-3 { border-left:5px solid #95d5b2; background:#fbfffc; }
.result-crop { font-size:1.15rem; font-weight:700; color:#163a2b; line-height:1.25; }
.result-conf { font-size:.78rem; color:#66736c; margin-top:.22rem; }
.conf-bar-wrap { background:#dfeee5; border-radius:20px; height:7px; margin:.65rem 0; overflow:hidden; }
.conf-bar { height:7px; border-radius:20px; background:linear-gradient(90deg,#74c69d,#2d6a4f); }
.advisory-box {
    background:#fffdf0; border:1px solid #eee6a8; border-radius:9px;
    padding:.68rem .85rem; font-size:.78rem; color:#5d5b4b;
    margin-top:.7rem; line-height:1.55;
}
.snapshot-wrap {
    background:#fafcfb; border:1px solid #e7eee9; border-radius:14px;
    padding:1rem; margin-top:.8rem;
}
.stat-row { display:flex; gap:.45rem; flex-wrap:wrap; }
.stat-pill {
    background:#f0faf4; border:1px solid #cfe8d8; border-radius:20px;
    padding:.3rem .72rem; font-size:.73rem; color:#2d6a4f; font-weight:500;
}
.empty-state {
    background:#f8fffb; border:1px dashed #b7dcc5; border-radius:14px;
    padding:2.1rem 1rem; text-align:center; color:#666; margin-top:1rem;
}
.disclaimer {
    background:#f7f8f7; border:1px solid #e7e9e7; border-radius:10px;
    padding:.7rem .85rem; color:#6b6f6c; font-size:.72rem; line-height:1.5; margin-top:1rem;
}
.footer {
    text-align:center; color:#9a9f9b; font-size:.72rem; margin-top:2.3rem;
    padding-top:.9rem; border-top:1px solid #edf0ee;
}
@media (max-width:700px) {
    .block-container { padding-left:.8rem; padding-right:.8rem; }
    .header-wrap { padding:1.25rem; border-radius:16px; }
    .header-icon { width:50px; height:50px; font-size:1.75rem; }
    .header-title { font-size:1.45rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load("crop_model_final.pkl")

try:
    assets = load_assets()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.caption("Ensure crop_model_final.pkl is in the same directory as app.py.")
    st.exception(exc)
    st.stop()

st.markdown("""
<div class="header-wrap">
  <div class="header-icon">🌾</div>
  <div>
    <div class="header-title">KrishiDarshan</div>
    <div class="header-sub">Climate-Aware Crop Recommendation</div>
    <span class="header-badge">35,364 records · 47 crops · 4 states · 2011–2022</span>
  </div>
</div>
<div class="info-strip">
  <b>How it works:</b> Enter your location, climate, soil and farm conditions.
  KrishiDarshan analyses the supplied values and presents the three highest-scoring
  crop recommendations with crop-specific advisory information.
</div>
""", unsafe_allow_html=True)

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

with st.form("crop_form"):
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📍 Location & Season</div>', unsafe_allow_html=True)
        state = st.selectbox("State", assets["le_dict"]["state"].classes_)
        season = st.selectbox("Season", assets["le_dict"]["season"].classes_)
        month = st.selectbox("Month", list(range(1,13)), format_func=lambda x: months[x-1])
        soiltype = st.selectbox("Soil Type", assets["le_dict"]["soiltype"].classes_)
        croptype = st.selectbox("Crop Category", assets["le_dict"]["croptype"].classes_)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🌦 Climate Conditions</div>', unsafe_allow_html=True)
        temp = st.slider("Temperature (°C)", 5.0, 50.0, 28.0, .5)
        rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 120.0, 5.0)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 70.0, 1.0)
        ph = st.slider("Soil pH", 4.0, 9.5, 6.5, .1)
        moisture = st.slider("Soil Moisture (0–1)", 0.0, 1.0, .4, .05)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🧪 Soil Nutrients & Farm</div>', unsafe_allow_html=True)
        n = st.number_input("Nitrogen — N (kg/ha)", 0, 200, 90)
        p = st.number_input("Phosphorus — P (kg/ha)", 0, 200, 45)
        k = st.number_input("Potassium — K (kg/ha)", 0, 200, 55)
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 500.0, 100.0)
        area = st.number_input("Farm Area (ha)", 1.0, 10000.0, 100.0)
        yield_val = st.number_input("Expected Yield (T/ha)", .1, 30.0, 2.5)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌱  Analyse & Recommend Crop", use_container_width=True)

if submitted:
    try:
        with st.spinner("Analysing your field conditions…"):
            inp = {
                "temperature(c)": float(temp),
                "tempanomaly(c)": 0.0,
                "rainfall(mm)": float(rainfall),
                "humidity(%)": float(humidity),
                "soilph": float(ph),
                "soilmoisture": float(moisture),
                "soiltype": int(assets["le_dict"]["soiltype"].transform([soiltype])[0]),
                "n": float(n), "p": float(p), "k": float(k),
                "fertilizerconsumption(kg/ha)": float(fertilizer),
                "month": int(month),
                "season": int(assets["le_dict"]["season"].transform([season])[0]),
                "state": int(assets["le_dict"]["state"].transform([state])[0]),
                "croptype": int(assets["le_dict"]["croptype"].transform([croptype])[0]),
                "yield": float(yield_val), "area": float(area),
            }
            row_df = pd.DataFrame([inp])[assets["features"]].astype(float)
            row_scaled = assets["scaler"].transform(row_df)
            probs = assets["model"].predict_proba(row_scaled)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        labels = ["Best Match", "2nd Choice", "3rd Choice"]
        cards = ["result-rank-1", "result-rank-2", "result-rank-3"]
        medals = ["🥇", "🥈", "🥉"]

        st.markdown("### Recommended Crops")
        st.caption("Ranked using the classifier's predict_proba() output. Scores are not calibrated probabilities of field success.")

        for rank, idx in enumerate(top3_idx):
            crop = assets["le_crop"].classes_[idx]
            score = float(probs[idx] * 100)
            advisory = assets["advisory"].get(crop, "Follow recommended agronomic practices for this crop.")
            st.markdown(f"""
<div class="result-card {cards[rank]}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
    <div>
      <div class="result-crop">{crop}</div>
      <div class="result-conf">{labels[rank]} · {score:.1f}% predicted score</div>
    </div>
    <div style="font-size:1.45rem;">{medals[rank]}</div>
  </div>
  <div class="conf-bar-wrap"><div class="conf-bar" style="width:{min(max(score,0),100):.1f}%;"></div></div>
  <div class="advisory-box"><b>Farming Advisory</b><br>{advisory[:350]}{"…" if len(advisory)>350 else ""}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="snapshot-wrap">
  <div style="font-weight:700;color:#1b4332;margin-bottom:.65rem;">Your Input Snapshot</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="stat-row">
  <span class="stat-pill">📍 {state}</span>
  <span class="stat-pill">🗓 {season} · {months[month-1]}</span>
  <span class="stat-pill">🌡 {temp:.1f}°C</span>
  <span class="stat-pill">🌧 {rainfall:.0f} mm</span>
  <span class="stat-pill">💧 {humidity:.0f}%</span>
  <span class="stat-pill">🧪 pH {ph:.1f}</span>
  <span class="stat-pill">🌿 N{n:.0f} / P{p:.0f} / K{k:.0f}</span>
  <span class="stat-pill">🪨 {soiltype}</span>
  <span class="stat-pill">🌾 {croptype}</span>
</div></div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="disclaimer">
  <b>Decision-support note:</b> Predicted scores are outputs of the trained machine-learning
  model. They are not guarantees of crop yield, profitability, or field performance. Final
  decisions should also consider local agronomic advice and current field conditions.
</div>
""", unsafe_allow_html=True)

    except Exception as exc:
        st.error("The prediction could not be completed.")
        st.caption("Check the supplied values and confirm that crop_model_final.pkl matches the application.")
        st.exception(exc)
else:
    st.markdown("""
<div class="empty-state">
  <div style="font-size:2.4rem;margin-bottom:.4rem;">🌱</div>
  <div style="font-weight:700;color:#2d6a4f;font-size:1rem;">Ready to analyse your field?</div>
  <div style="font-size:.82rem;margin-top:.35rem;color:#888;">
    Enter location, climate, soil and farm information, then select <b>Analyse & Recommend Crop</b>.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <b>KrishiDarshan</b> · Climate-Aware Crop Recommendation<br>
  Kushal Mohan · Jayesh Sharma · Anirudh Singh · Remant Jha
</div>
""", unsafe_allow_html=True)
    background:#f4fbf7; border:1px solid #d8eee0; border-radius:12px;
    padding:.8rem 1rem; margin-bottom:1rem; color:#315b47; font-size:.84rem; line-height:1.55;
}
.section-label {
    font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#2d6a4f; margin:.2rem 0 .7rem; padding-bottom:.45rem; border-bottom:1px solid #e1efe6;
}
.form-card {
    background:#fff; border:1px solid #e2eee6; border-radius:14px;
    padding:1.05rem 1.15rem .65rem; min-height:100%;
    box-shadow:0 3px 14px rgba(20,60,40,.035);
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius:9px; }
div[data-testid="stFormSubmitButton"] button {
    min-height:48px; background:#2d6a4f !important; color:white !important;
    border:none !important; border-radius:11px !important; font-weight:700 !important;
    font-size:.98rem !important; box-shadow:0 5px 15px rgba(45,106,79,.18);
}
div[data-testid="stFormSubmitButton"] button:hover { background:#1b4332 !important; transform:translateY(-1px); }
.result-card {
    border-radius:14px; padding:1rem 1.2rem; margin-bottom:.8rem;
    border:1px solid #e5ece7; box-shadow:0 3px 14px rgba(20,60,40,.04);
}
.result-rank-1 { border-left:5px solid #2d6a4f; background:#f0faf4; }
.result-rank-2 { border-left:5px solid #52b788; background:#f6fcf8; }
.result-rank-3 { border-left:5px solid #95d5b2; background:#fbfffc; }
.result-crop { font-size:1.15rem; font-weight:700; color:#163a2b; line-height:1.25; }
.result-conf { font-size:.78rem; color:#66736c; margin-top:.22rem; }
.conf-bar-wrap { background:#dfeee5; border-radius:20px; height:7px; margin:.65rem 0; overflow:hidden; }
.conf-bar { height:7px; border-radius:20px; background:linear-gradient(90deg,#74c69d,#2d6a4f); }
.advisory-box {
    background:#fffdf0; border:1px solid #eee6a8; border-radius:9px;
    padding:.68rem .85rem; font-size:.78rem; color:#5d5b4b;
    margin-top:.7rem; line-height:1.55;
}
.snapshot-wrap {
    background:#fafcfb; border:1px solid #e7eee9; border-radius:14px;
    padding:1rem; margin-top:.8rem;
}
.stat-row { display:flex; gap:.45rem; flex-wrap:wrap; }
.stat-pill {
    background:#f0faf4; border:1px solid #cfe8d8; border-radius:20px;
    padding:.3rem .72rem; font-size:.73rem; color:#2d6a4f; font-weight:500;
}
.empty-state {
    background:#f8fffb; border:1px dashed #b7dcc5; border-radius:14px;
    padding:2.1rem 1rem; text-align:center; color:#666; margin-top:1rem;
}
.disclaimer {
    background:#f7f8f7; border:1px solid #e7e9e7; border-radius:10px;
    padding:.7rem .85rem; color:#6b6f6c; font-size:.72rem; line-height:1.5; margin-top:1rem;
}
.footer {
    text-align:center; color:#9a9f9b; font-size:.72rem; margin-top:2.3rem;
    padding-top:.9rem; border-top:1px solid #edf0ee;
}
@media (max-width:700px) {
    .block-container { padding-left:.8rem; padding-right:.8rem; }
    .header-wrap { padding:1.25rem; border-radius:16px; }
    .header-icon { width:50px; height:50px; font-size:1.75rem; }
    .header-title { font-size:1.45rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load("crop_model_final.pkl")

try:
    assets = load_assets()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.caption("Ensure crop_model_final.pkl is in the same directory as app.py.")
    st.exception(exc)
    st.stop()

st.markdown("""
<div class="header-wrap">
  <div class="header-icon">🌾</div>
  <div>
    <div class="header-title">KrishiDarshan</div>
    <div class="header-sub">Climate-Aware Crop Recommendation</div>
    <span class="header-badge">35,364 records · 47 crops · 4 states · 2011–2022</span>
  </div>
</div>
<div class="info-strip">
  <b>How it works:</b> Enter your location, climate, soil and farm conditions.
  KrishiDarshan analyses the supplied values and presents the three highest-scoring
  crop recommendations with crop-specific advisory information.
</div>
""", unsafe_allow_html=True)

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

with st.form("crop_form"):
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📍 Location & Season</div>', unsafe_allow_html=True)
        state = st.selectbox("State", assets["le_dict"]["state"].classes_)
        season = st.selectbox("Season", assets["le_dict"]["season"].classes_)
        month = st.selectbox("Month", list(range(1,13)), format_func=lambda x: months[x-1])
        soiltype = st.selectbox("Soil Type", assets["le_dict"]["soiltype"].classes_)
        croptype = st.selectbox("Crop Category", assets["le_dict"]["croptype"].classes_)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🌦 Climate Conditions</div>', unsafe_allow_html=True)
        temp = st.slider("Temperature (°C)", 5.0, 50.0, 28.0, .5)
        rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 120.0, 5.0)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 70.0, 1.0)
        ph = st.slider("Soil pH", 4.0, 9.5, 6.5, .1)
        moisture = st.slider("Soil Moisture (0–1)", 0.0, 1.0, .4, .05)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🧪 Soil Nutrients & Farm</div>', unsafe_allow_html=True)
        n = st.number_input("Nitrogen — N (kg/ha)", 0, 200, 90)
        p = st.number_input("Phosphorus — P (kg/ha)", 0, 200, 45)
        k = st.number_input("Potassium — K (kg/ha)", 0, 200, 55)
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 500.0, 100.0)
        area = st.number_input("Farm Area (ha)", 1.0, 10000.0, 100.0)
        yield_val = st.number_input("Expected Yield (T/ha)", .1, 30.0, 2.5)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌱  Analyse & Recommend Crop", use_container_width=True)

if submitted:
    try:
        with st.spinner("Analysing your field conditions…"):
            inp = {
                "temperature(c)": float(temp),
                "tempanomaly(c)": 0.0,
                "rainfall(mm)": float(rainfall),
                "humidity(%)": float(humidity),
                "soilph": float(ph),
                "soilmoisture": float(moisture),
                "soiltype": int(assets["le_dict"]["soiltype"].transform([soiltype])[0]),
                "n": float(n), "p": float(p), "k": float(k),
                "fertilizerconsumption(kg/ha)": float(fertilizer),
                "month": int(month),
                "season": int(assets["le_dict"]["season"].transform([season])[0]),
                "state": int(assets["le_dict"]["state"].transform([state])[0]),
                "croptype": int(assets["le_dict"]["croptype"].transform([croptype])[0]),
                "yield": float(yield_val), "area": float(area),
            }
            row_df = pd.DataFrame([inp])[assets["features"]].astype(float)
            row_scaled = assets["scaler"].transform(row_df)
            probs = assets["model"].predict_proba(row_scaled)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        labels = ["Best Match", "2nd Choice", "3rd Choice"]
        cards = ["result-rank-1", "result-rank-2", "result-rank-3"]
        medals = ["🥇", "🥈", "🥉"]

        st.markdown("### Recommended Crops")
        st.caption("Ranked using the classifier's predict_proba() output. Scores are not calibrated probabilities of field success.")

        for rank, idx in enumerate(top3_idx):
            crop = assets["le_crop"].classes_[idx]
            score = float(probs[idx] * 100)
            advisory = assets["advisory"].get(crop, "Follow recommended agronomic practices for this crop.")
            st.markdown(f"""
<div class="result-card {cards[rank]}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
    <div>
      <div class="result-crop">{crop}</div>
      <div class="result-conf">{labels[rank]} · {score:.1f}% predicted score</div>
    </div>
    <div style="font-size:1.45rem;">{medals[rank]}</div>
  </div>
  <div class="conf-bar-wrap"><div class="conf-bar" style="width:{min(max(score,0),100):.1f}%;"></div></div>
  <div class="advisory-box"><b>Farming Advisory</b><br>{advisory[:350]}{"…" if len(advisory)>350 else ""}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="snapshot-wrap">
  <div style="font-weight:700;color:#1b4332;margin-bottom:.65rem;">Your Input Snapshot</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="stat-row">
  <span class="stat-pill">📍 {state}</span>
  <span class="stat-pill">🗓 {season} · {months[month-1]}</span>
  <span class="stat-pill">🌡 {temp:.1f}°C</span>
  <span class="stat-pill">🌧 {rainfall:.0f} mm</span>
  <span class="stat-pill">💧 {humidity:.0f}%</span>
  <span class="stat-pill">🧪 pH {ph:.1f}</span>
  <span class="stat-pill">🌿 N{n:.0f} / P{p:.0f} / K{k:.0f}</span>
  <span class="stat-pill">🪨 {soiltype}</span>
  <span class="stat-pill">🌾 {croptype}</span>
</div></div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="disclaimer">
  <b>Decision-support note:</b> Predicted scores are outputs of the trained machine-learning
  model. They are not guarantees of crop yield, profitability, or field performance. Final
  decisions should also consider local agronomic advice and current field conditions.
</div>
""", unsafe_allow_html=True)

    except Exception as exc:
        st.error("The prediction could not be completed.")
        st.caption("Check the supplied values and confirm that crop_model_final.pkl matches the application.")
        st.exception(exc)
else:
    st.markdown("""
<div class="empty-state">
  <div style="font-size:2.4rem;margin-bottom:.4rem;">🌱</div>
  <div style="font-weight:700;color:#2d6a4f;font-size:1rem;">Ready to analyse your field?</div>
  <div style="font-size:.82rem;margin-top:.35rem;color:#888;">
    Enter location, climate, soil and farm information, then select <b>Analyse & Recommend Crop</b>.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <b>KrishiDarshan</b> · Climate-Aware Crop Recommendation<br>
  Kushal Mohan · Jayesh Sharma · Anirudh Singh · Remant Jha
</div>
""", unsafe_allow_html=True)
"Enter the soil-moisture value expected by the trained model.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🧪 Soil Nutrients & Farm</div>', unsafe_allow_html=True)

        n = st.number_input(
            "Nitrogen — N (kg/ha)", 0, 200, 90,
            help="Nitrogen value supplied to the trained model.",
        )
        p = st.number_input(
            "Phosphorus — P (kg/ha)", 0, 200, 45,
            help="Phosphorus value supplied to the trained model.",
        )
        k = st.number_input(
            "Potassium — K (kg/ha)", 0, 200, 55,
            help="Potassium value supplied to the trained model.",
        )
        fertilizer = st.number_input(
            "Fertilizer (kg/ha)", 0.0, 500.0, 100.0,
            help="Fertilizer consumption value supplied to the trained model.",
        )
        area = st.number_input(
            "Farm Area (ha)", 1.0, 10000.0, 100.0,
            help="Farm area supplied to the trained model.",
        )
        yield_val = st.number_input(
            "Expected Yield (T/ha)", 0.1, 30.0, 2.5,
            help="Expected yield value supplied to the trained model.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🌱  Analyse & Recommend Crop",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
if submitted:
    try:
        with st.spinner("Analysing your field conditions…"):
            # The deployed model expects temperature anomaly as a model feature.
            # The current application supplies 0.0 because there is no user-facing
            # temperature-anomaly field.
            inp = {
                "temperature(c)": float(temp),
                "tempanomaly(c)": 0.0,
                "rainfall(mm)": float(rainfall),
                "humidity(%)": float(humidity),
                "soilph": float(ph),
                "soilmoisture": float(moisture),
                "soiltype": int(
                    assets["le_dict"]["soiltype"].transform([soiltype])[0]
                ),
                "n": float(n),
                "p": float(p),
                "k": float(k),
                "fertilizerconsumption(kg/ha)": float(fertilizer),
                "month": int(month),
                "season": int(
                    assets["le_dict"]["season"].transform([season])[0]
                ),
                "state": int(
                    assets["le_dict"]["state"].transform([state])[0]
                ),
                "croptype": int(
                    assets["le_dict"]["croptype"].transform([croptype])[0]
                ),
                "yield": float(yield_val),
                "area": float(area),
            }

            # Preserve the exact feature order stored with the trained model.
            row_df = pd.DataFrame([inp])[assets["features"]].astype(float)
            row_scaled = assets["scaler"].transform(row_df)
            probs = assets["model"].predict_proba(row_scaled)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        rank_class = ["result-rank-1", "result-rank-2", "result-rank-3"]
        rank_label = ["Best Match", "2nd Choice", "3rd Choice"]
        medals = ["🥇", "🥈", "🥉"]

        st.markdown(
            """
            <div class="results-heading">
                <h3 style="margin-bottom:0.1rem;">Recommended Crops</h3>
                <div class="results-sub">
                    Ranked using the model's <i>predict_proba()</i> output.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for rank, idx in enumerate(top3_idx):
            crop = assets["le_crop"].classes_[idx]
            score = float(probs[idx] * 100)
            adv = assets["advisory"].get(
                crop,
                "Follow recommended agronomic practices for this crop."
            )
            bar = min(max(score, 0.0), 100.0)

            st.markdown(
                f"""
<div class="result-card {rank_class[rank]}">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
        <div>
            <div class="result-crop">{crop}</div>
            <div class="result-conf">{rank_label[rank]} · {score:.1f}% predicted score</div>
        </div>
        <div style="font-size:1.45rem;">{medals[rank]}</div>
    </div>
    <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:{bar:.1f}%;"></div>
    </div>
    <div class="advisory-box">
        <b>Farming Advisory</b><br>
        {adv[:350]}{"…" if len(adv) > 350 else ""}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Input recap
        st.markdown(
            """
            <div class="snapshot-wrap">
                <div style="font-weight:700; color:#1b4332; margin-bottom:0.65rem;">
                    Your Input Snapshot
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="stat-row">
                <span class="stat-pill">📍 {state}</span>
                <span class="stat-pill">🗓 {season} · {month_names[month-1]}</span>
                <span class="stat-pill">🌡 {temp:.1f}°C</span>
                <span class="stat-pill">🌧 {rainfall:.0f} mm</span>
                <span class="stat-pill">💧 {humidity:.0f}%</span>
                <span class="stat-pill">🧪 pH {ph:.1f}</span>
                <span class="stat-pill">🌿 N{n:.0f} / P{p:.0f} / K{k:.0f}</span>
                <span class="stat-pill">🪨 {soiltype}</span>
                <span class="stat-pill">🌾 {croptype}</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="disclaimer">
                <b>Decision-support note:</b> Predicted scores are outputs of the trained
                machine-learning model. They are not guarantees of crop yield, profitability,
                or field performance. Final crop decisions should also consider local
                agronomic advice and current field conditions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(
            "The prediction could not be completed. Please check the supplied values "
            "and confirm that crop_model_final.pkl matches the current application."
        )
        st.exception(exc)

else:
    st.markdown(
        """
<div class="empty-state">
    <div class="empty-icon">🌱</div>
    <div class="empty-title">Ready to analyse your field?</div>
    <div class="empty-copy">
        Enter location, climate, soil and farm information above,
        then select <b>Analyse & Recommend Crop</b>.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    <b>KrishiDarshan</b> · Climate-Aware Crop Recommendation<br>
    Kushal Mohan · Jayesh Sharma · Anirudh Singh · Remant Jha
</div>
""",
    unsafe_allow_html=True,
)
