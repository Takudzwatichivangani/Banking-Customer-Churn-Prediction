"""ETL pipeline nodes with validation and logging."""
import pandas as pd
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# ✅ ADDED: Input validation function
def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate raw data schema and quality."""
    required_columns = {
        "CreditScore", "Geography", "Gender", "Age", "Tenure", 
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
        "EstimatedSalary", "Exited"
    }
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # Check for nulls in critical columns
    critical_cols = ["CreditScore", "Age", "Exited"]
    null_counts = df[critical_cols].isnull().sum()
    if null_counts.any():
        return False, f"Null values in critical columns: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Validation passed"

# ✅ ADDED: Data quality metrics
def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data quality metrics for monitoring."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)),
        "duplicate_rows": df.duplicated().sum(),
        "churn_rate": df["Exited"].mean() if "Exited" in df.columns else None,
    }

# ✅ MODIFIED: Enhanced extract function with validation
def extract_raw(raw_churn: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract and validate raw data."""
    logger.info(f"Extracting raw data with shape: {raw_churn.shape}")
    
    # Create a copy
    df = raw_churn.copy()
    
    # Validate data
    is_valid, validation_message = validate_raw_data(df)
    if not is_valid:
        logger.error(f"Data validation failed: {validation_message}")
        raise ValueError(f"Invalid raw data: {validation_message}")
    
    # Calculate quality metrics
    quality_metrics = calculate_data_quality_metrics(df)
    logger.info(f"Data quality metrics: {quality_metrics}")
    
    return df, quality_metrics

# ✅ MODIFIED: Enhanced cleaning function
def clean_and_prepare_data(
    raw_df: pd.DataFrame, 
    quality_metrics: Dict[str, Any],
    drop_columns: list,
    random_state: int
) -> pd.DataFrame:
    """Clean and prepare data for modeling."""
    df = raw_df.copy()
    logger.info(f"Starting data cleaning with shape: {df.shape}")
    
    # Drop specified columns
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Dropped columns: {columns_to_drop}")
    
    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate rows")
    
    # Ensure proper data types
    if "Exited" in df.columns:
        df["Exited"] = df["Exited"].astype(int)
    
    # Log final shape
    logger.info(f"Cleaned data shape: {df.shape}")
    
    return df