# src/preprocessing.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config


def identify_feature_types(X):
    """
    Automatically categorizes columns into numeric and categorical features.
    
    Args:
        X (pd.DataFrame): Features dataframe
    
    Returns:
        tuple: (num_cols, cat_cols) - lists of column names
    """
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features ({len(num_cols)}): {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")
    
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, 
                       numeric_strategy=None,
                       categorical_strategy=None,
                       categorical_fill_value=None,
                       handle_unknown=None):
    """
    Builds a sklearn ColumnTransformer preprocessing pipeline.
    
    Pipeline structure:
    - Numeric: Impute (median/mean) -> StandardScaler
    - Categorical: Impute (constant/most_frequent) -> OneHotEncoder
    
    Args:
        num_cols (list): List of numeric column names
        cat_cols (list): List of categorical column names
        numeric_strategy (str, optional): Imputation strategy for numeric. Defaults to config.
        categorical_strategy (str, optional): Imputation strategy for categorical. Defaults to config.
        categorical_fill_value (str, optional): Fill value for categorical. Defaults to config.
        handle_unknown (str, optional): How to handle unknown categories. Defaults to config.
    
    Returns:
        sklearn.compose.ColumnTransformer: Fitted preprocessing pipeline
    """
    if numeric_strategy is None:
        numeric_strategy = config.NUMERIC_IMPUTE_STRATEGY
    if categorical_strategy is None:
        categorical_strategy = config.CATEGORICAL_IMPUTE_STRATEGY
    if categorical_fill_value is None:
        categorical_fill_value = config.CATEGORICAL_IMPUTE_FILL_VALUE
    if handle_unknown is None:
        handle_unknown = config.HANDLE_UNKNOWN
    
    # Numeric pipeline: Impute missing with training statistic, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing, then One-Hot Encode
    # handle_unknown='ignore' ensures pipeline doesn't break if production data
    # introduces a new category not seen in training
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value=categorical_fill_value)),
        ('onehot', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
    ])
    
    # Combine into single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'  # Drops any columns not explicitly defined above
    )
    
    return preprocessor


def fit_and_transform(preprocessor, X_train, X_test):
    """
    Fits the preprocessor on training data and transforms both train and test sets.
    
    Args:
        preprocessor: sklearn ColumnTransformer
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_processed, X_test_processed, feature_names)
    """
    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Retrieve feature names for downstream interpretability
    feature_names = preprocessor.get_feature_names_out()
    print(f"✅ Resulting feature count: {len(feature_names)}")
    
    return X_train_processed, X_test_processed, feature_names


def save_processed_data(X_train_processed, X_test_processed, 
                       y_train, y_test, 
                       feature_names, 
                       target_col=None,
                       train_path=None,
                       test_path=None):
    """
    Saves processed data to CSV files with feature names and target.
    
    Args:
        X_train_processed (np.ndarray): Processed training features
        X_test_processed (np.ndarray): Processed test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        feature_names (list): Feature column names
        target_col (str, optional): Target column name. Defaults to config.TARGET_COLUMN.
        train_path (str, optional): Training data save path. Defaults to config.PROCESSED_TRAIN_PATH.
        test_path (str, optional): Test data save path. Defaults to config.PROCESSED_TEST_PATH.
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    if train_path is None:
        train_path = config.PROCESSED_TRAIN_PATH
    if test_path is None:
        test_path = config.PROCESSED_TEST_PATH
    
    # Ensure directories exist
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    # Re-attach target variables and save to CSVs
    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df[target_col] = y_train.reset_index(drop=True)
    
    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df[target_col] = y_test.reset_index(drop=True)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Processed data saved:")
    print(f"   Training: {train_path}")
    print(f"   Testing: {test_path}")


def save_preprocessor(preprocessor, path=None):
    """
    Saves the fitted preprocessor pipeline to disk using joblib.
    
    Args:
        preprocessor: sklearn ColumnTransformer
        path (str, optional): Save path. Defaults to config.PREPROCESSOR_PATH.
    """
    if path is None:
        path = config.PREPROCESSOR_PATH
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"✅ Preprocessor saved: {path}")


def load_preprocessor(path=None):
    """
    Loads a fitted preprocessor pipeline from disk.
    
    Args:
        path (str, optional): Load path. Defaults to config.PREPROCESSOR_PATH.
    
    Returns:
        sklearn.compose.ColumnTransformer: Loaded preprocessor
    """
    if path is None:
        path = config.PREPROCESSOR_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Preprocessor not found at {path}")
    
    preprocessor = joblib.load(path)
    print(f"✅ Preprocessor loaded from: {path}")
    return preprocessor


def preprocess_pipeline(X_train, X_test, y_train, y_test, 
                       save_artifacts=True):
    """
    End-to-end preprocessing pipeline that:
    1. Identifies feature types
    2. Builds preprocessor
    3. Fits and transforms data
    4. Saves artifacts (optional)
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        save_artifacts (bool, optional): Whether to save processed data and preprocessor. Defaults to True.
    
    Returns:
        tuple: (X_train_processed, X_test_processed, feature_names, preprocessor)
    """
    # Step 1: Identify feature types
    num_cols, cat_cols = identify_feature_types(X_train)
    
    # Step 2: Build preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # Step 3: Fit and transform
    X_train_processed, X_test_processed, feature_names = fit_and_transform(
        preprocessor, X_train, X_test
    )
    
    # Step 4: Save artifacts
    if save_artifacts:
        save_processed_data(X_train_processed, X_test_processed, 
                          y_train, y_test, feature_names)
        save_preprocessor(preprocessor)
    
    print("✅ Preprocessing pipeline complete!")
    return X_train_processed, X_test_processed, feature_names, preprocessor


if __name__ == "__main__":
    # Test the preprocessing functions
    from src.data_loader import load_raw_data, validate_and_clean_target, split_data
    
    try:
        print("Loading raw data...")
        df = load_raw_data()
        
        print("\nValidating target...")
        df = validate_and_clean_target(df)
        
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = split_data(df)
        
        print("\nRunning preprocessing pipeline...")
        X_train_processed, X_test_processed, feature_names, preprocessor = preprocess_pipeline(
            X_train, X_test, y_train, y_test
        )
        
        print(f"\nFinal shapes:")
        print(f"X_train_processed: {X_train_processed.shape}")
        print(f"X_test_processed: {X_test_processed.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")