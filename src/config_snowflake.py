from snowflake.snowpark.context import get_active_session
import joblib
import os

MLFLOW_URI = "sqlite:////tmp/mlflow.db"

def load_data():
    session = get_active_session()
    # Reads directly from the Snowflake secure stage
    return session.read.parquet("@ML_DATA_STAGE/integrated_features.parquet").to_pandas()

def save_artifacts(preprocessor, model, run_name):
    session = get_active_session()
    os.makedirs('/tmp/models', exist_ok=True)
    joblib.dump(preprocessor, f"/tmp/models/preprocessor_{run_name}.joblib")
    joblib.dump(model, f"/tmp/models/model_{run_name}.joblib")
    
    # Securely push the ephemeral disk files into permanent Snowflake storage
    session.sql("CREATE STAGE IF NOT EXISTS @ML_ARTIFACTS_STAGE").collect()
    session.file.put("file:///tmp/mlflow.db", "@ML_ARTIFACTS_STAGE/mlflow/", auto_compress=False, overwrite=True)
    session.file.put(f"file:///tmp/models/*.joblib", "@ML_ARTIFACTS_STAGE/models/", auto_compress=False, overwrite=True)
    print("✅ Artifacts secured in Snowflake Stage.")