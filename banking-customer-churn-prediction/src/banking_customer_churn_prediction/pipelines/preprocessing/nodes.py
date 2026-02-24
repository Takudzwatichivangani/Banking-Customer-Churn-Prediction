"""Data preprocessing nodes."""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ✅ ENHANCED: Build preprocessor with validation
def build_preprocessor(
    numeric_cols: list,
    categorical_cols: list,
    preprocessing_params: Dict
) -> ColumnTransformer:
    """Build preprocessing pipeline with configurable strategies."""
    
    numeric_params = preprocessing_params.get("numeric", {})
    categorical_params = preprocessing_params.get("categorical", {})
    
    # Numeric pipeline
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=numeric_params.get("impute_strategy", "median"))),
        ("scale", StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(
            strategy=categorical_params.get("impute_strategy", "most_frequent"),
            fill_value="missing"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", 
            sparse_output=False,
            drop="first"  # ✅ ADDED: Avoid dummy variable trap
        ))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")  # ✅ ADDED: Drop unused columns
    
    logger.info(f"Built preprocessor for {len(numeric_cols)} numeric "
                f"and {len(categorical_cols)} categorical features")
    
    return preprocessor

# ✅ NEW: Fit preprocessor on training data only
def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessor on training data and transform all splits."""
    
    logger.info("Fitting preprocessor on training data...")
    
    # Fit on training data only
    preprocessor.fit(X_train)
    
    # Transform all splits
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name != "remainder":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_trans, columns=feature_names, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)
    
    logger.info(f"Preprocessing complete. Feature count: {len(feature_names)}")
    
    return X_train_df, X_val_df, X_test_df, preprocessor