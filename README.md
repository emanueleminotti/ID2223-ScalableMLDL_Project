# 🌿 Pollen Forecast & Allergy Risk Service  
**ID2223 – Scalable Machine Learning Systems (HT2025)**  
*End-to-End Pollen Forecasting with Feature Stores, ML Pipelines and Interactive UI*

---

## 🌍 Live Application
The interactive Streamlit dashboard is available at:

👉 **https://id2223-scalablemldlproject-ad7hddhzjnhjmedrzqia6d.streamlit.app/**

The application starts in **Live Data mode** and allows switching to **Demo (Placeholder) mode** for presentation purposes.

---

## Overview

This project implements an **end-to-end machine learning system** for forecasting **airborne pollen levels** and estimating **allergy risk** for the next **7 days**.

The system covers the full ML lifecycle:
- data ingestion and feature engineering  
- feature storage using a Feature Store  
- model training and registration  
- batch inference  
- visualization through an interactive **Streamlit dashboard**

The project is developed as part of the course  
**ID2223 – Scalable Machine Learning Systems**, following best practices from *Building ML Systems with a Feature Store*.

---

## Forecasted Pollens

The system forecasts daily concentration levels for the following pollen types:

- **Alder pollen**
- **Birch pollen**
- **Grass pollen**
- **Mugwort pollen**

Each pollen type is modeled within a **multi-output regression framework**.

---

## Key Features

### ✅ 7-Day Pollen Forecast  
For each pollen type, the system produces:
- daily predicted pollen levels  
- a derived **risk level** (Low / Medium / High)  
- a normalized **risk score (0–100)**  

### ✅ Allergy Risk Interpretation  
Predicted pollen levels are mapped to **human-interpretable risk categories** using pollen-specific thresholds inspired by aerobiological guidelines and clinical practice.

The UI visualizes:
- background risk bands  
- threshold lines (Low → Medium → High)  
- daily health suggestions  

---

## System Architecture

The project is structured as **four main pipelines**, implemented as Jupyter notebooks.

### 1️⃣ Feature Backfill Pipeline
**`1_feature_backfill.ipynb`**
- Loads historical pollen and meteorological data  
- Writes features to the **Hopsworks Feature Store**  

### 2️⃣ Daily Feature Pipeline
**`2_feature_pipeline.ipynb`**
- Runs daily  
- Updates lagged and rolling features  

### 3️⃣ Training Pipeline
**`3_training_pipeline.ipynb`**
- Builds a Feature View  
- Trains a **Multi-Output XGBoost Regressor**  
- Evaluates with R² and RMSE  
- Registers the model in the **Hopsworks Model Registry**  

### 4️⃣ Batch Inference Pipeline
**`4_inference_pipeline.ipynb`**
- Loads the latest model  
- Generates 7-day forecasts  
- Writes outputs as CSV and JSON  
- Optionally stores predictions in a monitoring Feature Group  

---

## Streamlit Dashboard

The Streamlit app provides an interactive visualization layer:

- Tabs per pollen type  
- Time-series plot with risk bands and thresholds  
- Clickable daily selection  
- Detailed daily risk explanation and suggestions  

### Live vs Demo Mode
- **Live mode (default):** real model predictions  
- **Demo mode:** synthetic but realistic placeholder data  

This ensures the application remains demonstrable even outside pollen season.

---

## Project Structure

```
ID2223-ScalableMLDL_Project/
│
├── app.py
├── notebooks/
│   ├── 1_feature_backfill.ipynb
│   ├── 2_feature_pipeline.ipynb
│   ├── 3_training_pipeline.ipynb
│   ├── 4_inference_pipeline.ipynb
│   └── latest_forecasts.json
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python  
- Hopsworks Feature Store & Model Registry  
- XGBoost  
- Pandas / NumPy  
- Streamlit  
- Matplotlib  
- GitHub Actions  

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Authors

**Emanuele Minotti** – minotti@kth.se  
**Stefano Romano** – sromano@kth.se  

Group project for  
**ID2223 – Scalable Machine Learning Systems (HT2025)**  
KTH Royal Institute of Technology
