# 🌾 SmartKISHAN: AI-Powered Crop Recommendation System

**SmartKISHAN** is a data-driven agriculture tool designed to help farmers maximize their yield. By analyzing soil nutrients and local weather conditions, the app uses Machine Learning to suggest the most suitable crops for a specific farm.

## 🚀 Live Demo
[Insert your Streamlit Cloud Link Here]

## 🌟 Key Features
* **XGBoost Engine:** High-precision predictions based on historical agricultural data.
* **Top-3 Recommendations:** Provides multiple options with confidence percentages.
* **Smart Advisory:** Specific farming tips and warnings for each suggested crop.
* **Responsive Design:** Easy to use on both mobile and desktop.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Joblib

## 📊 How it Works
1. **Input Data:** Enter soil parameters (N, P, K, pH) and weather data (Temperature, Rainfall, Humidity).
2. **Preprocessing:** The app scales the data using a pre-trained `StandardScaler`.
3. **Prediction:** The XGBoost model calculates the probability for various crops.
4. **Result:** The system displays the top 3 crops along with professional advisory notes.

## 📁 Repository Structure
* `app.py`: The main web application script.
* `crop_model_final.pkl`: The saved model, scaler, and encoders.
* `requirements.txt`: List of required Python libraries.
* `.gitignore`: Files to be ignored by Git.

---
Developed with ❤️ for the farming community.
