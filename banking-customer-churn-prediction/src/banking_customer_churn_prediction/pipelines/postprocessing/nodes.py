"""Postprocessing nodes for formatting and saving results."""
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def write_high_risk_to_csv(high_risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format and prepare high-risk customers for CSV export.
    Returns the formatted DataFrame.
    """
    logger.info(f"Preparing {len(high_risk_df)} high-risk customers for export")
    
    # Create a copy to avoid modifying the original
    df = high_risk_df.copy()
    
    # Ensure we have the required columns
    required_columns = []
    
    # Add timestamp if not present
    if "inference_timestamp" not in df.columns:
        df["export_timestamp"] = datetime.utcnow().isoformat()
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "probability" in col.lower() or "confidence" in col.lower():
            df[col] = df[col].round(4)
            # Add percentage version
            df[f"{col}_percent"] = (df[col] * 100).round(2)
    
    # Sort by risk (highest probability first)
    if "probability_churn" in df.columns:
        df = df.sort_values("probability_churn", ascending=False)
    
    logger.info(f"Formatted {len(df)} rows for export")
    
    return df

def format_business_output(
    predictions_df: pd.DataFrame,
    original_features: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Format predictions for business consumption.
    """
    business_output = predictions_df.copy()
    
    # Merge with original features if available
    if original_features is not None and not original_features.empty:
        # Try to merge on index
        try:
            business_output = pd.merge(
                business_output,
                original_features,
                left_index=True,
                right_index=True,
                how="left",
                suffixes=("", "_original")
            )
        except:
            logger.warning("Could not merge with original features")
    
    # Rename columns for readability
    column_mapping = {
        "prediction": "Predicted_Churn",
        "probability_churn": "Churn_Probability",
        "risk_category": "Risk_Category",
        "will_churn": "Will_Churn_Flag",
        "recommendations": "Recommended_Actions",
        "confidence": "Prediction_Confidence"
    }
    
    # Apply renaming for columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in business_output.columns:
            business_output = business_output.rename(columns={old_name: new_name})
    
    # Add export metadata
    business_output["Export_Timestamp"] = datetime.utcnow().isoformat()
    business_output["Export_Version"] = "1.0.0"
    
    return business_output

def generate_summary_report(
    high_risk_df: pd.DataFrame,
    stats: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate summary statistics for high-risk customers.
    """
    df = high_risk_df.copy()
    
    summary = {
        "report_date": datetime.utcnow().isoformat(),
        "total_customers": len(df),
        "high_risk_count": 0,
        "statistics": {}
    }
    
    # Count high-risk customers
    if "risk_category" in df.columns:
        summary["high_risk_count"] = len(df[df["risk_category"] == "high_risk"])
        summary["risk_distribution"] = df["risk_category"].value_counts().to_dict()
    
    # Calculate average churn probability
    if "probability_churn" in df.columns:
        summary["statistics"]["average_churn_probability"] = float(df["probability_churn"].mean())
        summary["statistics"]["max_churn_probability"] = float(df["probability_churn"].max())
        summary["statistics"]["min_churn_probability"] = float(df["probability_churn"].min())
    
    # Add provided stats if available
    if stats:
        summary["statistics"].update(stats)
    
    logger.info(f"Generated summary report: {summary['high_risk_count']} high-risk customers")
    
    return summary

def save_to_multiple_formats(
    df: pd.DataFrame,
    base_path: str = "data/07_model_output"
) -> Dict[str, str]:
    """
    Save DataFrame to multiple formats (CSV, Parquet, JSON).
    Returns paths to saved files.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_path) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    try:
        # Save as CSV
        csv_path = base_dir / "high_risk_customers.csv"
        df.to_csv(csv_path, index=False)
        saved_files["csv"] = str(csv_path)
        logger.info(f"Saved CSV to: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
    
    try:
        # Save as Parquet
        parquet_path = base_dir / "high_risk_customers.parquet"
        df.to_parquet(parquet_path, index=False)
        saved_files["parquet"] = str(parquet_path)
        logger.info(f"Saved Parquet to: {parquet_path}")
    except Exception as e:
        logger.error(f"Failed to save Parquet: {e}")
    
    try:
        # Save as JSON (first 100 rows)
        json_path = base_dir / "high_risk_customers_sample.json"
        sample_df = df.head(100)
        sample_df.to_json(json_path, orient="records", indent=2)
        saved_files["json"] = str(json_path)
        logger.info(f"Saved JSON sample to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
    
    # Save summary
    summary = {
        "export_timestamp": timestamp,
        "total_rows": len(df),
        "file_paths": saved_files,
        "columns": list(df.columns)
    }
    
    summary_path = base_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    saved_files["summary"] = str(summary_path)
    
    return saved_files