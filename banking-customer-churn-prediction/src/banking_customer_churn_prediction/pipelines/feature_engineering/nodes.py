"""Feature engineering pipeline nodes."""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ✅ NEW: Comprehensive feature engineering
def create_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    feature_params: Dict
) -> pd.DataFrame:
    """Create engineered features from cleaned data."""
    features_df = df.copy()
    
    # Create derived features
    derived_features = feature_params.get("derived", [])
    
    for feat_config in derived_features:
        feat_name = feat_config["name"]
        
        if feat_name == "avg_bal_per_product":
            if "Balance" in features_df.columns and "NumOfProducts" in features_df.columns:
                features_df[feat_name] = (
                    features_df["Balance"] / (features_df["NumOfProducts"] + 1e-9)
                )
                logger.info(f"Created feature: {feat_name}")
                
        elif feat_name == "tenure_bucket" and "Tenure" in features_df.columns:
            bins = feat_config.get("bins", [-1, 1, 3, 6, 10, 50])
            features_df[feat_name] = pd.cut(
                features_df["Tenure"], 
                bins=bins, 
                labels=False
            )
            logger.info(f"Created feature: {feat_name}")
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str)
    
    logger.info(f"Feature engineering complete. Shape: {features_df.shape}")
    return features_df

# ✅ NEW: Split data with stratification
def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool
) -> Dict[str, pd.DataFrame]:
    """
    Split data into features and target, then into train/val/test.
    Returns dictionary of datasets.
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train+val vs test
    stratify_y = y if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )
    
    # Second split: train vs val
    if stratify:
        stratify_train_val = y_train_val
    else:
        stratify_train_val = None
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_train_val
    )
    
    # Log split statistics
    logger.info(f"Data split complete:")
    logger.info(f"  Train: {X_train.shape[0]} samples ({len(y_train[y_train==1])} positive)")
    logger.info(f"  Val: {X_val.shape[0]} samples ({len(y_val[y_val==1])} positive)")
    logger.info(f"  Test: {X_test.shape[0]} samples ({len(y_test[y_test==1])} positive)")
    
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test
    }