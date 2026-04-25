import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Assets (Keep this at the top)
@st.cache_resource
def load_assets():
    return joblib.load('crop_model_final.pkl')

assets = load_assets()

# 2. Define the UI and Form
st.title("SmartKISHAN")

with st.form("my_form"):
    # ... (all your input fields like state, temp, etc.) ...
    state = st.selectbox("State", assets['le_dict']['state'].classes_)
    # ...
    
    # THIS LINE CREATES THE VARIABLE 'submitted'
    submitted = st.form_submit_button("Get Recommendations")

# 3. Prediction Logic (THIS MUST BE AFTER THE FORM)
if submitted:
    # Your prediction code goes here...
    st.write("Calculating...")
