"""Model and data monitoring nodes."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from scipy.stats import ks_2samp, chi2_contingency
import json

logger = logging.getLogger(__name__)

# ✅ NEW: Comprehensive drift detection
def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect data drift between reference and current data.
    Returns drift report with p-values and alerts.
    """
    drift_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "columns_checked": [],
        "drift_detected": False,
        "alerts": [],
        "metrics": {}
    }
    
    threshold = drift_params.get("threshold", 0.05)
    sample_size = drift_params.get("sample_size", 1000)
    
    # Sample data if too large
    if len(reference_data) > sample_size:
        reference_sample = reference_data.sample(sample_size, random_state=42)
    else:
        reference_sample = reference_data
    
    if len(current_data) > sample_size:
        current_sample = current_data.sample(sample_size, random_state=42)
    else:
        current_sample = current_data
    
    # Check numeric columns
    numeric_cols = reference_sample.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in current_sample.columns:
            try:
                stat, p_value = ks_2samp(
                    reference_sample[col].dropna(),
                    current_sample[col].dropna()
                )
                
                drift_report["columns_checked"].append(col)
                drift_report["metrics"][col] = {
                    "p_value": float(p_value),
                    "statistic": float(stat),
                    "drift_detected": p_value < threshold
                }
                
                if p_value < threshold:
                    drift_report["drift_detected"] = True
                    alert_msg = f"Drift detected in {col}: p={p_value:.4f}"
                    drift_report["alerts"].append(alert_msg)
                    logger.warning(alert_msg)
                    
            except Exception as e:
                logger.error(f"Error checking drift for {col}: {e}")
    
    # Check categorical columns
    categorical_cols = reference_sample.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in current_sample.columns:
            try:
                # Create contingency table
                ref_counts = reference_sample[col].value_counts()
                curr_counts = current_sample[col].value_counts()
                
                # Align categories
                all_cats = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_cats]
                
                # Chi-square test
                chi2, p_value, dof, expected = chi2_contingency(
                    [ref_aligned, curr_aligned]
                )
                
                drift_report["columns_checked"].append(col)
                drift_report["metrics"][col] = {
                    "p_value": float(p_value),
                    "chi2": float(chi2),
                    "drift_detected": p_value < threshold
                }
                
                if p_value < threshold:
                    drift_report["drift_detected"] = True
                    alert_msg = f"Categorical drift in {col}: p={p_value:.4f}"
                    drift_report["alerts"].append(alert_msg)
                    logger.warning(alert_msg)
                    
            except Exception as e:
                logger.error(f"Error checking categorical drift for {col}: {e}")
    
    # Calculate overall drift score
    if drift_report["metrics"]:
        p_values = [m["p_value"] for m in drift_report["metrics"].values()]
        drift_report["overall_drift_score"] = np.mean(p_values)
    
    logger.info(f"Drift detection complete. Drift detected: {drift_report['drift_detected']}")
    
    return drift_report

# ✅ NEW: Model performance monitoring
def monitor_model_performance(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    reference_metrics: Dict[str, float],
    monitoring_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Monitor model performance degradation.
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return {"error": "No data for performance monitoring"}
    
    # Align data
    aligned_idx = predictions.index.intersection(actuals.index)
    if len(aligned_idx) == 0:
        return {"error": "No overlapping indices"}
    
    preds_aligned = predictions.loc[aligned_idx]
    actuals_aligned = actuals.loc[aligned_idx]
    
    # Calculate current metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    current_metrics = {
        "roc_auc": roc_auc_score(actuals_aligned, preds_aligned["probability_churn"]),
        "accuracy": accuracy_score(actuals_aligned, preds_aligned["prediction"]),
        "sample_size": len(aligned_idx),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check for degradation
    degradation_report = {
        "current_metrics": current_metrics,
        "reference_metrics": reference_metrics,
        "degradation_detected": False,
        "degradation_alerts": []
    }
    
    threshold = monitoring_params.get("retrain_threshold", 0.02)
    
    for metric_name, current_value in current_metrics.items():
        if metric_name in reference_metrics and isinstance(current_value, (int, float)):
            reference_value = reference_metrics[metric_name]
            degradation = reference_value - current_value
            
            degradation_report[f"{metric_name}_degradation"] = degradation
            
            if degradation > threshold:
                degradation_report["degradation_detected"] = True
                alert_msg = f"Performance degradation in {metric_name}: {degradation:.4f}"
                degradation_report["degradation_alerts"].append(alert_msg)
                logger.warning(alert_msg)
    
    return degradation_report

# ✅ NEW: Generate monitoring dashboard data
def generate_monitoring_dashboard(
    drift_report: Dict[str, Any],
    performance_report: Dict[str, Any],
    risk_statistics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate consolidated data for monitoring dashboard.
    """
    dashboard_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_quality": {
            "drift_detected": drift_report.get("drift_detected", False),
            "drift_alerts": drift_report.get("alerts", []),
            "overall_drift_score": drift_report.get("overall_drift_score", 1.0)
        },
        "model_performance": {
            "degradation_detected": performance_report.get("degradation_detected", False),
            "current_metrics": performance_report.get("current_metrics", {}),
            "degradation_alerts": performance_report.get("degradation_alerts", [])
        },
        "business_metrics": risk_statistics,
        "alerts": drift_report.get("alerts", []) + performance_report.get("degradation_alerts", [])
    }
    
    # Calculate overall health score
    health_score = 100
    
    if dashboard_data["data_quality"]["drift_detected"]:
        health_score -= 20
    
    if dashboard_data["model_performance"]["degradation_detected"]:
        health_score -= 30
    
    dashboard_data["system_health_score"] = max(0, health_score)
    dashboard_data["system_status"] = (
        "healthy" if health_score >= 80 else
        "warning" if health_score >= 60 else
        "critical"
    )
    
    logger.info(f"Monitoring dashboard generated. Status: {dashboard_data['system_status']}")
    
    return dashboard_data