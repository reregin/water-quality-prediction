# 🌊 Water Quality Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A machine learning project to predict water quality parameters across river locations in South Africa, with emphasis on identifying key factors influencing water quality variation.

**Project Objective:**
Develop a robust ML model capable of predicting water quality parameters (total alkalinity, electrical conductance, dissolved reactive phosphorus) and identify the key environmental and geographic factors that significantly influence these measurements.

---

## 📂 Project Structure

This project follows a strict separation of concerns.

```
├── data/
│   ├── raw/                  # Original water quality dataset (2011-2015, ~200 locations)
│   ├── processed/            # Cleaned & feature-engineered data
│   └── external/             # Geographic/environmental reference data
│
├── notebooks/                              # Experimental Laboratory
│   ├── 00_data_collection.ipynb            # Data loading & exploration
│   ├── 01_eda_and_discovery.ipynb          # Discovery & Analysis (Split-First: Train Only)
│   ├── 02_preprocessing.ipynb              # Feature Engineering & Transformation
│   ├── 03_model_training.ipynb             # Model Training with MLflow Tracking
│   └── 04_inference_test.ipynb             # Validation & Feature Importance Analysis
│
├── src/                      # Production Codebase
│   ├── config.py             # Global Control Center (Paths, Params)
│   ├── data_loader.py        # Robust Data Ingestion & Splitting
│   ├── preprocessing.py      # Reusable Cleaning & Feature Engineering Logic
│   ├── train.py              # Model Training Pipeline
│   ├── inference.py          # Prediction Engine
│   └── utils.py              # Helper Functions
│
├── models/                   # Serialized Models (.pkl, .pth)
├── app/
│   └── main.py               # User Interface (Streamlit/FastAPI)
├── mlflow.db                 # MLflow Experiment Tracking Database
└── requirements.txt          # Dependencies
```

---

## 📊 Dataset Overview

- **Time Period**: 2011 - 2015
- **Sampling Sites**: ~200 river locations across South Africa
- **Water Quality Parameters**: Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus
- **Features**: Geographic coordinates (latitude/longitude), sampling date