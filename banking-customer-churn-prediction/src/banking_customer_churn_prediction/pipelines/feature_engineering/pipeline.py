"""Feature engineering pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import create_features, split_features_target

def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=create_features,
                inputs=[
                    "cleaned_customers",
                    "params:data.columns.numeric",
                    "params:data.columns.categorical",
                    "params:features"
                ],
                outputs="features_df",
                name="create_engineered_features",
                tags=["feature_engineering"],
            ),
            node(
                func=split_features_target,
                inputs=[
                    "features_df",
                    "params:data.target",
                    "params:data.test_size",
                    "params:data.val_size",
                    "params:data.random_state",
                    "params:data.stratify"
                ],
                outputs={
                    "X_train": "X_train",
                    "X_val": "X_val", 
                    "X_test": "X_test",
                    "y_train": "y_train",
                    "y_val": "y_val", 
                    "y_test": "y_test"
                },
                name="split_features_target",
                tags=["feature_engineering", "splitting"],
            ),
        ]
    )