# src/train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings

# Sklearn Imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# MLflow (Optional - Handle import error if not installed)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Training will proceed without experiment tracking.")

# XGBoost (Optional - Handle import error if not installed)
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

# LightGBM (Optional)
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None

# Dynamic Path Setup to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config


# ==========================================
# 1. MODEL FACTORY
# ==========================================
def get_model(model_name, task_type="classification", **kwargs):
    """
    Returns an un-trained model instance based on the name in config.
    
    Args:
        model_name (str): Name of the model
        task_type (str): "classification" or "regression"
        **kwargs: Additional model parameters
    
    Returns:
        Sklearn-compatible model instance
    """
    print(f"🔧 Initializing model: {model_name} ({task_type})")
    
    # Merge default random state with custom kwargs
    if 'random_state' not in kwargs:
        kwargs['random_state'] = config.RANDOM_STATE
    
    # --- CLASSIFICATION MODELS ---
    if task_type == "classification":
        if model_name == "random_forest":
            return RandomForestClassifier(n_estimators=100, **kwargs)
        elif model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, **kwargs)
        elif model_name == "svm":
            return SVC(probability=True, **kwargs)
        elif model_name == "xgboost" and XGBClassifier:
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
        elif model_name == "lightgbm" and LGBMClassifier:
            return LGBMClassifier(**kwargs)
            
    # --- REGRESSION MODELS ---
    elif task_type == "regression":
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=100, **kwargs)
        elif model_name == "linear_regression":
            kwargs.pop('random_state', None)  # LinearRegression doesn't use random_state
            return LinearRegression(**kwargs)
        elif model_name == "xgboost" and XGBRegressor:
            return XGBRegressor(**kwargs)
        elif model_name == "lightgbm" and LGBMRegressor:
            return LGBMRegressor(**kwargs)

    raise ValueError(f"❌ Model '{model_name}' not supported for task '{task_type}'")


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_processed_data(train_path=None, test_path=None, target_col=None):
    """
    Loads preprocessed training and test data, then splits into X and y.
    
    Args:
        train_path (str, optional): Path to training data. Defaults to config.PROCESSED_TRAIN_PATH.
        test_path (str, optional): Path to test data. Defaults to config.PROCESSED_TEST_PATH.
        target_col (str, optional): Target column name. Defaults to config.TARGET_COLUMN.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if train_path is None:
        train_path = config.PROCESSED_TRAIN_PATH
    if test_path is None:
        test_path = config.PROCESSED_TEST_PATH
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Training data not found at {train_path}. Run preprocessing first!")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Test data not found at {test_path}. Run preprocessing first!")
    
    print(f"📂 Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"📂 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Split features and target
    if target_col not in train_df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in training data.")
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test shapes:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test


# ==========================================
# 3. METRICS CALCULATION
# ==========================================
def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add ROC-AUC for binary classification
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            pass
    
    return metrics


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    
    return metrics


# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Creates and optionally saves a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str, optional): Path to save the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plots feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n (int): Number of top features to display
        save_path (str, optional): Path to save the figure
    
    Returns:
        tuple: (figure, importance_dataframe)
    """
    if not hasattr(model, 'feature_importances_'):
        return None, None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_data = importance_df.head(top_n).sort_values('importance', ascending=True)
    ax.barh(plot_data['feature'], plot_data['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Feature importance plot saved to: {save_path}")
    
    return fig, importance_df


# ==========================================
# 5. TRAINING PIPELINE
# ==========================================
def train_model(model_name=None, 
                task_type=None, 
                model_params=None,
                run_name="Baseline_Model",
                log_mlflow=True,
                experiment_name="DS_Template_Project"):
    """
    End-to-end training pipeline with MLflow tracking.
    
    Args:
        model_name (str, optional): Model name. Defaults to config.MODEL_NAME.
        task_type (str, optional): Task type. Defaults to config.TASK_TYPE.
        model_params (dict, optional): Model hyperparameters. Defaults to {}.
        run_name (str): MLflow run name
        log_mlflow (bool): Whether to use MLflow tracking
        experiment_name (str): MLflow experiment name
    
    Returns:
        tuple: (trained_model, metrics)
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    if task_type is None:
        task_type = config.TASK_TYPE
    if model_params is None:
        model_params = {}
    
    # Setup MLflow if available and requested
    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"sqlite:///{config.PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment(experiment_name)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Initialize model
    model = get_model(model_name, task_type, **model_params)
    
    # Start MLflow run if available
    if log_mlflow and MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=run_name):
            metrics = _train_and_log(model, X_train, X_test, y_train, y_test, 
                                    model_name, task_type, model_params)
    else:
        metrics = _train_and_log(model, X_train, X_test, y_train, y_test,
                                model_name, task_type, model_params, log_mlflow=False)
    
    return model, metrics


def _train_and_log(model, X_train, X_test, y_train, y_test,
                   model_name, task_type, model_params, log_mlflow=True):
    """
    Internal function to train model and log to MLflow.
    """
    # Log parameters
    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_params(model_params)
        mlflow.log_param("input_rows", X_train.shape[0])
        mlflow.log_param("input_cols", X_train.shape[1])
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_state", config.RANDOM_STATE)
    
    # Train model
    print("🚀 Training started...")
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # Make predictions
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics based on task type
    if task_type == "classification":
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') and len(np.unique(y_train)) == 2 else None
        metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        
        # Print classification report
        print("\n" + classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm_path = os.path.join(config.MODELS_DIR, 'confusion_matrix.png')
        fig_cm = plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
        
        if log_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)
        
    else:  # regression
        metrics = calculate_regression_metrics(y_test, y_pred)
    
    # Log metrics
    print(f"\n📈 Metrics: {metrics}")
    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics)
    
    # Feature importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        importance_path = os.path.join(config.MODELS_DIR, 'feature_importance.png')
        fig_imp, importance_df = plot_feature_importance(model, X_train.columns, save_path=importance_path)
        
        if fig_imp and log_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_figure(fig_imp, "feature_importance.png")
            
            # Save importance CSV
            csv_path = os.path.join(config.MODELS_DIR, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path='feature_importance')
        
        if fig_imp:
            plt.close(fig_imp)
    
    # Save model
    model_path = config.MODEL_PATH
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Log model to MLflow
    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.sklearn.log_model(model, "model")
        print(f"\n✅ Run complete! Check MLflow UI: mlflow ui --backend-store-uri sqlite:///{config.PROJECT_ROOT}/mlflow.db")
    
    return metrics


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Example: Train with default config settings
    model, metrics = train_model(
        run_name="EXP_01_Baseline",
        experiment_name="DS_Template_Project"
    )
    
    print("\n" + "=" * 50)
    print("🎉 Training pipeline completed successfully!")
    print("=" * 50)