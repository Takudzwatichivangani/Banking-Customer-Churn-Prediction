"""Inference pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import batch_inference, flag_high_risk_customers, generate_recommendations

def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline."""
    return Pipeline(
        [
            node(
                func=batch_inference,
                inputs=[
                    "best_model",
                    "X_test_processed",
                    "feature_preprocessor_fitted",
                    "params:inference.include_confidence"
                ],
                outputs="predictions_raw",
                name="batch_inference",
                tags=["inference", "batch"],
            ),
            node(
                func=flag_high_risk_customers,
                inputs=[
                    "predictions_raw",
                    "params:inference"
                ],
                outputs=["predictions_flagged", "risk_statistics"],
                name="flag_high_risk_customers",
                tags=["inference", "risk_scoring"],
            ),
            node(
                func=generate_recommendations,
                inputs=[
                    "predictions_flagged",
                    "X_test",
                    "params:recommendation_rules"
                ],
                outputs="predictions_with_recommendations",
                name="generate_recommendations",
                tags=["inference", "business_logic"],
            ),
        ]
    )