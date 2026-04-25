# KrishiDarshan — Climate-Aware Crop Recommendation System

A machine learning project built as part of the **Design Thinking & Innovation (DTI)** course at Manav Rachna University. The system analyses soil nutrients, climate conditions, and seasonal patterns to recommend the most suitable crops for a given region in India.


## Acknowledgements

We thank our project mentor **Mrs. Swati Hans** (School of Engineering & Technology, Manav Rachna University) for her continuous guidance and feedback throughout the DTI project lifecycle.

---
**Live App →** [smart-kishan.streamlit.app](https://smart-kishan.streamlit.app)

---

## Screenshots


*Input form — location, climate sliders, and soil nutrients*
<img width="1904" height="924" alt="Screenshot 2026-04-25 183505" src="https://github.com/user-attachments/assets/ff54e6f9-bc45-491d-8094-59d395d41b06" />
)

*Top-3 crop recommendations with confidence scores and farming advisory*
<img width="1808" height="808" alt="Screenshot 2026-04-25 183753" src="https://github.com/user-attachments/assets/d292f048-d94f-40cd-af2c-100d79b8d5cf" />


---

## What it does

- Takes soil parameters (N, P, K, pH, moisture) and climate data (temperature, rainfall, humidity) as inputs
- Runs a trained Random Forest classifier to predict the best-fit crops
- Returns the top 3 crops with confidence percentages and crop-specific farming advisories
- Covers **47 crops**, **6 Indian states**, data from **2011–2025** (35,364 records)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Web App | Streamlit |
| ML Model | Random Forest, Scikit-Learn |
| Data | Pandas, NumPy |
| Model Save | Joblib |
| Data Source | Govt. of India — IMD, CPCB, NBSS&LUP |

---

## How it works

```
User Input (soil + climate)
        ↓
StandardScaler  →  normalise values
        ↓
Random Forest   →  predict probability for each crop
        ↓
Top-3 crops with confidence % + farming advisory
```

1. User fills in their farm details — state, season, soil type, N/P/K, temperature, rainfall
2. Values are encoded and scaled using the same pipeline trained on the dataset
3. The model outputs a probability for each of the 47 crops
4. Top 3 are shown with confidence bars and advisory notes pulled from the dataset

---

## Repository Structure

```
smartkishan/
├── app.py                           # Streamlit web application
├── crop_model_final.pkl             # Trained model + scaler + encoders
├── clean_datasets_with_advisory.csv # Dataset used for training
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/gitcoder20043328/smartkishan.git
cd smartkishan
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

Built from Government of India open data sources:

- **IMD** — monthly temperature, rainfall, humidity by district
- **CPCB** — soil quality indicators
- **NBSS&LUP** — soil type and nutrient data
- **data.gov.in** — crop production records

Covers Punjab, Haryana, Uttar Pradesh, Bihar, Rajasthan, and Madhya Pradesh from 2011 to 2025.

---

## Team

| Name | Roll No |
|---|---|
| Kushal | 1/24/SET/BCS/509 |
| Jayesh Sharma |  1/24/SET/BCS/512|
| Anirudh Singh |  1/24/SET/BCS/513 |
| Remant Jha    |  1/24/SET/BCS/518 |

**Institution:** Manav Rachna International Institute of Research and Studies  
