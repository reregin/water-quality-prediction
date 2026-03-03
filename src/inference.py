# src/inference.py
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config
from src import preprocessing as pp

# Global cache for the model (so we don't reload it 100 times)
_MODEL = None

def load_model():
    """
    Loads the trained model from the models/ directory.
    Uses a global variable to cache it (Singleton pattern).
    """
    global _MODEL
    
    if _MODEL is None:
        model_path = os.path.join("models", f"{config.MODEL_NAME}_v1.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Run src/train.py first!")
            
        print(f"Loading model from {model_path}...")
        _MODEL = joblib.load(model_path)
        
    return _MODEL

def make_prediction(input_data):
    """
    Main entry point for the app.
    Args:
        input_data (dict or pd.DataFrame): Raw input data.
    Returns:
        dict: {'prediction': int, 'probability': float}
    """
    # 1. Convert Dictionary to DataFrame (if needed)
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
        
    # 2. Preprocessing (Must match training!)
    # NOTE: In a real complex project, we would load a saved 'pipeline.pkl' here.
    # For this template, we apply the stateless cleaning functions.
    df = pp.clean_column_names(df)
    
    # 3. Load Model
    model = load_model()
    
    # 4. Predict
    # Ensure columns match model expectation
    try:
        prediction = model.predict(df)[0]
        
        # Get probability if supported (for "Risk Score")
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0][1] # Probability of Class 1 (Default)
        else:
            probability = None
            
        return {
            "prediction": int(prediction),
            "probability": float(probability) if probability else 0.0,
            "status": "Success"
        }
        
    except Exception as e:
        return {
            "prediction": None,
            "error": str(e),
            "status": "Failed"
        }

if __name__ == "__main__":
    # Test with dummy data
    dummy_data = {
        "limit_bal": 50000,
        "age": 35,
        "bill_amt1": 1200,
        "pay_amt1": 0
        # Add other columns as per dataset...
    }
    print(make_prediction(dummy_data))