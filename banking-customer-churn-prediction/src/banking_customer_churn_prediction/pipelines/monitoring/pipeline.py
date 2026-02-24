"""Monitoring pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import detect_data_drift, monitor_model_performance, generate_monitoring_dashboard

def create_pipeline(**kwargs) -> Pipeline:
    """Create the monitoring pipeline."""
    return Pipeline(
        [
            node(
                func=detect_data_drift,
                inputs=[
                    "X_train",  # Reference data
                    "X_test",   # Current/production data
                    "params:monitoring"
                ],
                outputs="data_drift_report",
                name="detect_data_drift",
                tags=["monitoring", "drift_detection"],
            ),
            node(
                func=monitor_model_performance,
                inputs=[
                    "predictions_with_recommendations",
                    "y_test",  # Ground truth for test set
                    "best_model_metrics",  # Reference metrics
                    "params:monitoring"
                ],
                outputs="performance_monitoring_report",
                name="monitor_model_performance",
                tags=["monitoring", "performance"],
            ),
            node(
                func=generate_monitoring_dashboard,
                inputs=[
                    "data_drift_report",
                    "performance_monitoring_report",
                    "risk_statistics"
                ],
                outputs="monitoring_dashboard_data",
                name="generate_monitoring_dashboard",
                tags=["monitoring", "dashboard"],
            ),
        ]
    )