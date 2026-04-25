if submitted:
    try:
        # Prepare Input - Note the [0] at the end of transform calls
        inp = {
            'temperature(c)': float(temp), 
            'tempanomaly(c)': 0.0, 
            'rainfall(mm)': float(rainfall),
            'humidity(%)': float(humidity), 
            'soilph': float(ph), 
            'soilmoisture': 0.4,
            
            # Extract the first element [0] to get a plain number
            'soiltype': assets['le_dict']['soiltype'].transform([soil_type]),
            'n': int(n), 
            'p': int(p), 
            'k': int(k), 
            'fertilizerconsumption(kg/ha)': 100.0,
            'month': 6, 
            'season': assets['le_dict']['season'].transform([season]),
            'state': assets['le_dict']['state'].transform([state]),
            'croptype': assets['le_dict']['croptype'].transform([croptype]),
            'yield': 2.5, 
            'area': float(area),
        }
        
        # Ensure columns match the original FEATURES list exactly
        row_df = pd.DataFrame([inp])[assets['features']]
        
        # Scale the numeric data
        row_scaled = assets['scaler'].transform(row_df)
        
        # Predict
        probs = assets['model'].predict_proba(row_scaled)[0] # Extract first row
        
        # Show Results
        top3_idx = np.argsort(probs)[::-1][:3]
        
        st.subheader("🌾 Top Recommendations")
        for i, idx in enumerate(top3_idx):
            crop = assets['le_crop'].classes_[idx]
            conf = probs[idx] * 100
            if conf > 0.01:
                st.success(f"**#{i+1}: {crop}** ({conf:.1f}% Confidence)")
                st.info(f"📋 **Advisory:** {assets['advisory'].get(crop, 'Standard practices apply.')}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
