# src/data_loader.py
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config


def load_raw_data(path=None):
    """
    Loads the raw data from the path specified in config.py.
    Handles CSV, Excel, and Parquet automatically.
    
    Args:
        path (str, optional): Path to the data file. Defaults to config.RAW_DATA_PATH.
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if path is None:
        path = config.RAW_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Error: Data file not found at {path}. Check src/config.py")

    print(f"Loading data from: {path}...")
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    elif ext == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"❌ Unsupported file format: {ext}")

    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df


def validate_and_clean_target(df, target_col=None):
    """
    Validates that the target column exists and drops rows with missing targets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str, optional): Target column name. Defaults to config.TARGET_COLUMN.
    
    Returns:
        pd.DataFrame: Dataframe with missing targets removed
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataframe. Available columns: {list(df.columns)}")
    
    initial_shape = df.shape[0]
    df = df.dropna(subset=[target_col])
    dropped = initial_shape - df.shape[0]
    
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing target values")
    
    return df


def split_data(df, target_col=None, test_size=None, random_state=None, stratify=True):
    """
    Splits the dataframe into train and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str, optional): Target column name. Defaults to config.TARGET_COLUMN.
        test_size (float, optional): Test set proportion. Defaults to config.TEST_SIZE.
        random_state (int, optional): Random seed. Defaults to config.RANDOM_STATE.
        stratify (bool, optional): Whether to stratify split for classification. Defaults to True.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify for classification tasks (categorical or small number of unique values)
    stratify_param = None
    if stratify and (y.dtype == 'object' or y.nunique() < 20):
        stratify_param = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def load_processed_data(train_path=None, test_path=None):
    """
    Loads processed training and test data from CSV files.
    
    Args:
        train_path (str, optional): Path to training data. Defaults to config.PROCESSED_TRAIN_PATH.
        test_path (str, optional): Path to test data. Defaults to config.PROCESSED_TEST_PATH.
    
    Returns:
        tuple: train_df, test_df
    """
    if train_path is None:
        train_path = config.PROCESSED_TRAIN_PATH
    if test_path is None:
        test_path = config.PROCESSED_TEST_PATH
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Test data not found at {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Processed data loaded!")
    print(f"   Training: {train_df.shape}")
    print(f"   Testing: {test_df.shape}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Test the functions if run directly
    try:
        df = load_raw_data()
        print("\nFirst few rows:")
        print(df.head())
        
        df = validate_and_clean_target(df)
        X_train, X_test, y_train, y_test = split_data(df)
        
    except Exception as e:
        print(e)