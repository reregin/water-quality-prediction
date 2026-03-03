# src/config.py
import os

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_FILE = "dataset.csv"  # CHANGE THIS
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", RAW_DATA_FILE)

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_fe.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "test_fe.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")

# --- TARGET VARIABLE ---
TARGET_COLUMN = "target"  # CHANGE THIS

# --- DATA SPLIT ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- PREPROCESSING ---
NUMERIC_IMPUTE_STRATEGY = "median"  # "median", "mean", "constant"
CATEGORICAL_IMPUTE_STRATEGY = "constant"  # "most_frequent", "constant"
CATEGORICAL_IMPUTE_FILL_VALUE = "missing"
HANDLE_UNKNOWN = "ignore"  # For OneHotEncoder: "ignore" or "error"

# --- MODEL SELECTION ---
MODEL_NAME = "random_forest"  # "random_forest", "logistic_regression", "svm", "xgboost" (classification); "linear_regression" (regression)
TASK_TYPE = "classification"  # "classification", "regression"