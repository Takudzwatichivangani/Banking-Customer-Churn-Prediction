"""Preprocessing pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import build_preprocessor, fit_preprocessor

def create_pipeline(**kwargs) -> Pipeline:
    """Create the preprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=build_preprocessor,
                inputs=[
                    "params:data.columns.numeric",
                    "params:data.columns.categorical",
                    "params:features.preprocessing"
                ],
                outputs="feature_preprocessor",
                name="build_preprocessor",
                tags=["preprocessing"],
            ),
            node(
                func=fit_preprocessor,
                inputs=[
                    "feature_preprocessor",
                    "X_train",
                    "X_val", 
                    "X_test"
                ],
                outputs=[
                    "X_train_processed",
                    "X_val_processed", 
                    "X_test_processed",
                    "feature_preprocessor_fitted"
                ],
                name="fit_and_transform_preprocessor",
                tags=["preprocessing", "fitting"],
            ),
        ]
    )