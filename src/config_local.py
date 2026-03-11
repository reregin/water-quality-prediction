import pandas as pd
import os
import joblib

MLFLOW_URI = "sqlite:///../mlflow.db"

def load_data():
    # Reads from your local laptop path
    return pd.read_parquet("../data/interim/master_train.parquet")

def save_artifacts(preprocessor, model, run_name):
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessor, f"../models/preprocessor_{run_name}.joblib")
    joblib.dump(model, f"../models/model_{run_name}.joblib")
    print("✅ Artifacts saved locally.")