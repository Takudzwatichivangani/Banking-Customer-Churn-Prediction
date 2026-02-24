"""API dependencies and utilities."""
import logging
import joblib
import mlflow
import pandas as pd
from typing import Optional
from pathlib import Path
"""API dependencies and utilities."""
import logging
import joblib
import mlflow
import pandas as pd
from typing import Optional
from pathlib import Path
import os
from fastapi import Header, HTTPException
import time
import importlib

logger = logging.getLogger(__name__)

# Get absolute paths relative to this file's location
API_DIR = Path(__file__).parent  # /src/banking_customer_churn_prediction/api
SRC_DIR = API_DIR.parent         # /src/banking_customer_churn_prediction
PROJECT_ROOT = SRC_DIR.parent.parent

# Configuration - absolute paths from project root
MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    str(PROJECT_ROOT / "data" / "06_models" / "best_model.pkl")
)
PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH", 
    str(PROJECT_ROOT / "data" / "04_feature" / "preprocessor.pkl")
)
METRICS_PATH = os.getenv(
    "METRICS_PATH", 
    str(PROJECT_ROOT / "data" / "06_models" / "model_metrics.json")
)
API_KEYS = os.getenv("API_KEYS", "test123,development").split(",")

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Preprocessor path: {PREPROCESSOR_PATH}")

# Cache for loaded models
_model_cache = None
_preprocessor_cache = None
_metrics_cache = None

def fit_preprocessor(preprocessor, save: bool = False):
    """Auto-fit an unfitted preprocessor with sample data.

    This helper is intended for local development only. By default it will
    NOT overwrite production artifacts unless `save=True` or the
    `AUTO_FIT_SAVE` env var is set to `true`.
    """
    logger.warning("Preprocessor not fitted. Auto-fitting with sample data...")

    sample_data = pd.DataFrame({
        'CustomerId': [100000 + i for i in range(6)],
        'CreditScore': [300, 450, 600, 750, 850, 500],
        'Geography': ['France', 'Germany', 'Spain', 'France', 'Germany', 'Spain'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Age': [18, 25, 35, 45, 65, 80],
        'Tenure': [0, 2, 5, 8, 10, 1],
        'Balance': [0.0, 50000.0, 150000.0, 250000.0, 0.0, 100000.0],
        'NumOfProducts': [1, 2, 3, 4, 1, 2],
        'HasCrCard': [0, 1, 0, 1, 0, 1],
        'IsActiveMember': [0, 1, 0, 1, 0, 1],
        'EstimatedSalary': [10000.0, 50000.0, 100000.0, 150000.0, 200000.0, 75000.0]
    })

    preprocessor.fit(sample_data)
    logger.info("✅ Preprocessor auto-fitted successfully")

    should_save = save or os.getenv("AUTO_FIT_SAVE", "false").lower() == "true"
    if should_save:
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logger.info(f"💾 Saved fitted preprocessor to: {PREPROCESSOR_PATH}")
    else:
        logger.info("Preprocessor auto-fitted in-memory (not saved). Set AUTO_FIT_SAVE=true to persist.)")

    return preprocessor

def get_model():
    """Get the trained model (cached)."""
    global _model_cache
    if _model_cache is None:
        try:
            # Try to load from MLflow first
            if os.getenv("MLFLOW_MODEL_URI"):
                _model_cache = mlflow.pyfunc.load_model(os.getenv("MLFLOW_MODEL_URI"))
                logger.info("Loaded model from MLflow")
            else:
                # Fallback to local file
                try:
                    _model_cache = joblib.load(MODEL_PATH)
                    logger.info(f"Loaded model from {MODEL_PATH}")
                except Exception as e_local:
                    logger.warning(f"Local model load failed: {e_local}")
                    # Try S3 fallback if configured
                    model_s3 = os.getenv("MODEL_S3_URI")
                    if model_s3:
                        logger.info(f"Attempting to download model from S3: {model_s3}")
                        dest = Path(MODEL_PATH)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            _download_model_from_s3(model_s3, str(dest))
                            _model_cache = joblib.load(MODEL_PATH)
                            logger.info(f"Loaded model from S3 into {MODEL_PATH}")
                        except Exception as e_s3:
                            logger.error(f"Failed to download or load model from S3: {e_s3}")
                            raise
                    else:
                        raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model_cache


def _download_model_from_s3(s3_uri: str, dest_path: str, attempts: int = 3):
    """Download an object from S3 to dest_path with simple retries.

    This implementation imports `boto3` at runtime so the module is optional
    for environments that don't need S3.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("MODEL_S3_URI must be an s3:// URI")
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError("Invalid s3 uri, expected s3://bucket/key")

    try:
        boto3 = importlib.import_module('boto3')
    except Exception as e:
        raise RuntimeError("boto3 is required to download models from S3") from e

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            s3 = boto3.client('s3')
            logger.info(f"Downloading s3://{bucket}/{key} -> {dest_path} (attempt {attempt})")
            s3.download_file(bucket, key, dest_path)
            return
        except Exception as e:
            last_exc = e
            sleep = min(2 ** attempt, 10)
            logger.warning(f"S3 download attempt {attempt} failed: {e}; retrying in {sleep}s")
            time.sleep(sleep)
    raise last_exc

def get_preprocessor():
    """Get preprocessor with auto-fitting if needed."""
    global _preprocessor_cache
    if _preprocessor_cache is None:
        try:
            logger.info(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
            _preprocessor_cache = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
            
            # Auto-detect and handle unfitted preprocessor
            if not hasattr(_preprocessor_cache, 'transformers_'):
                # In production we should fail fast; only auto-fit in non-production
                if os.getenv("ENV", "development") == "production":
                    logger.error("Preprocessor not fitted in production environment")
                    raise RuntimeError("Preprocessor artifact is not fitted")
                logger.warning("⚠️  Preprocessor is NOT fitted! Auto-fitting for development...")
                _preprocessor_cache = fit_preprocessor(_preprocessor_cache)
            else:
                logger.info("✅ Preprocessor is already fitted and ready")
                
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    return _preprocessor_cache

def verify_api_key(api_key: str) -> bool:
    """Verify API key."""
    # This function is kept for backwards compatibility when used as a
    # dependency. Prefer the header-based `x_api_key` signature below.
    if api_key in API_KEYS:
        return api_key
    logger.warning(f"Invalid API key attempt: {str(api_key)[:8]}...")
    raise HTTPException(status_code=401, detail="Invalid API key")


def verify_api_key_header(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """FastAPI dependency that verifies `X-API-Key` header.

    - In `production` the header is required and must match `API_KEYS`.
    - In non-production the header is optional (but validated if present).
    """
    env = os.getenv("ENV", "development")
    if env == "production":
        if not x_api_key or x_api_key not in API_KEYS:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
    else:
        if x_api_key and x_api_key not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not x_api_key:
            logger.debug("No API key provided (development mode)")
    return x_api_key