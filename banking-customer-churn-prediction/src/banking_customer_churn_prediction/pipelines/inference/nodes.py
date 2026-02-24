"""Inference pipeline nodes with monitoring."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ✅ ENHANCED: Batch inference with confidence scores
def batch_inference(
    model: Any,
    X: pd.DataFrame,
    preprocessor: Any,
    include_confidence: bool = True
) -> pd.DataFrame:
    """
    Run batch inference with preprocessing.
    Returns DataFrame with predictions and confidence scores.
    """
    logger.info(f"Running batch inference on {X.shape[0]} samples")
    
    # Preprocess
    X_processed = preprocessor.transform(X)
    
    # Predict
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)
    
    # Create results DataFrame
    results = pd.DataFrame({
        "prediction": predictions,
        "probability_churn": probabilities[:, 1],
        "probability_no_churn": probabilities[:, 0],
        "confidence": np.max(probabilities, axis=1),
        "inference_timestamp": datetime.utcnow().isoformat()
    }, index=X.index)
    
    # Add original index if available
    if hasattr(X, "index"):
        results["original_index"] = X.index
    
    logger.info(f"Inference complete. Churn rate: {predictions.mean():.2%}")
    
    return results

# ✅ ENHANCED: Risk flagging with multiple thresholds
def flag_high_risk_customers(
    predictions_df: pd.DataFrame,
    inference_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Flag high-risk customers using configurable thresholds.
    Returns flagged DataFrame and risk statistics.
    """
    df = predictions_df.copy()
    
    # Get thresholds
    churn_threshold = inference_params.get("threshold", 0.5)
    high_risk_threshold = inference_params.get("high_risk_threshold", 0.7)
    medium_risk_threshold = inference_params.get("medium_risk_threshold", 0.3)
    
    # Create risk categories
    conditions = [
        df["probability_churn"] >= high_risk_threshold,
        df["probability_churn"] >= churn_threshold,
        df["probability_churn"] >= medium_risk_threshold,
    ]
    choices = ["high_risk", "medium_risk", "low_risk"]
    df["risk_category"] = np.select(conditions, choices, default="very_low_risk")
    
    # Binary flag for churn
    df["will_churn"] = df["probability_churn"] >= churn_threshold
    
    # Calculate statistics
    stats = {
        "total_customers": len(df),
        "predicted_churn_rate": df["will_churn"].mean(),
        "high_risk_count": (df["risk_category"] == "high_risk").sum(),
        "medium_risk_count": (df["risk_category"] == "medium_risk").sum(),
        "low_risk_count": (df["risk_category"] == "low_risk").sum(),
        "very_low_risk_count": (df["risk_category"] == "very_low_risk").sum(),
        "average_churn_probability": df["probability_churn"].mean(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Risk categorization: {stats}")
    
    return df, stats

# ✅ NEW: Generate business recommendations
def generate_recommendations(
    high_risk_df: pd.DataFrame,
    original_data: pd.DataFrame,
    recommendation_rules: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate business recommendations for high-risk customers.
    """
    # Merge with original data for context
    results_df = high_risk_df.copy()
    if hasattr(original_data, "index"):
        results_df = pd.merge(
            results_df,
            original_data,
            left_index=True,
            right_index=True,
            how="left"
        )
    
    # Simple rule-based recommendations
    recommendations = []
    for idx, row in results_df.iterrows():
        rec = []
        
        if row.get("risk_category") == "high_risk":
            rec.append("Immediate retention call")
            if row.get("IsActiveMember", 0) == 0:
                rec.append("Activate membership benefits")
            if row.get("Balance", 0) > 100000:
                rec.append("Offer premium service consultation")
        
        elif row.get("risk_category") == "medium_risk":
            rec.append("Personalized email campaign")
            if row.get("NumOfProducts", 0) <= 1:
                rec.append("Cross-sell additional products")
        
        recommendations.append("; ".join(rec) if rec else "Monitor")
    
    results_df["recommendations"] = recommendations
    
    return results_df