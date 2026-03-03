# 🛡️ Data Science Project Template

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![Status](https://img.shields.io/badge/Status-Development-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A production-ready, modular structure for Data Science and Machine Learning projects. Designed to separate **experimentation** (notebooks) from **engineering** (src), ensuring reproducibility and scalability from Day 1.

**Key Features:**
- 🔬 **MLflow Integration** - Automated experiment tracking and model versioning
- 🛡️ **Data Leakage Prevention** - Split-first strategy ensuring test data never contaminates training
- 📦 **Modular Design** - Clean separation between research and production code

---

## 📂 Project Structure

This project follows a strict separation of concerns.

```text
├── data/
│   ├── raw/                  # Immutable original data (do not edit)
│   ├── processed/            # Cleaned data used for modeling
│   └── external/             # Third-party data/references
│
├── notebooks/                              # Experimental Laboratory
│   ├── 01_eda_and_discovery.ipynb          # Discovery & Analysis (Split-First: Train Only)
│   ├── 02_preprocessing.ipynb              # Data Cleaning & Transformation
│   ├── 03_model_training.ipynb             # Model Training with MLflow Tracking
│   └── 04_inference_test.ipynb             # Final Pipeline Validation
│
├── src/                      # Production Codebase
│   ├── config.py             # Global Control Center (Paths, Params)
│   ├── data_loader.py        # Robust Data Ingestion
│   ├── preprocessing.py      # Reusable Cleaning Logic
│   ├── train.py              # Model Training Pipeline
│   └── inference.py          # Prediction Engine (Singleton)
│
├── models/                   # Serialized Models (.pkl, .pth)
├── app/
│   └── main.py               # User Interface (Streamlit/FastAPI)
├── mlflow.db                 # MLflow Experiment Tracking Database
└── requirements.txt          # Dependencies
```

---

## 🔬 MLflow Experiment Tracking

This template integrates **MLflow** for comprehensive experiment tracking and model management.

### Setup
```python
import mlflow

# Set tracking URI (SQLite database)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment name
mlflow.set_experiment("YourProjectName")
```

### What Gets Tracked
- **Parameters**: Model hyperparameters, data split ratios, preprocessing steps
- **Metrics**: Accuracy, precision, recall, F1, RMSE, R², etc.
- **Artifacts**: Model files, plots, confusion matrices
- **Metadata**: Input shape, feature names, timestamps

### View Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Access the dashboard at `http://localhost:5000`

---

## 🛡️ Data Leakage Prevention Strategy

This template implements a **split-first approach** to eliminate data leakage risks:

### The Problem
Traditional workflows often perform EDA and preprocessing on the full dataset before splitting, which can lead to:
- Target leakage from feature engineering
- Scaling/imputation contaminated by test data statistics
- Overly optimistic model performance

### Our Solution: Split First, Always
1. **Load raw data** → Immediately split into train/test (80/20)
2. **EDA on train only** → All analysis, visualizations, and statistical summaries use training data exclusively
3. **Preprocessing fitted on train** → Transformers learn only from training data
4. **Test set remains untouched** → Locked away until final evaluation

### Implementation
```python
# ✅ Correct: Split FIRST
df = pd.read_csv('data.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Now use 'train' for all EDA and preprocessing
# 'test' is set aside until model evaluation
```

**Result**: Your model's test performance accurately reflects real-world generalization.

---

## 📦 Dependencies

Core libraries used in this template:

```
pandas
numpy
scikit-learn
mlflow
joblib
matplotlib
seaborn
streamlit
```

Install all dependencies:
```bash
pip install -r requirements.txt