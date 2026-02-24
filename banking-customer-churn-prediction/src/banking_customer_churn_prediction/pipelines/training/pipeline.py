"""Training pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import train_models_with_cv, log_to_mlflow, log_model_simple

def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline."""
    return Pipeline(
        [
            node(
                func=train_models_with_cv,
                inputs=[
                    "X_train_processed",
                    "y_train",
                    "X_val_processed",
                    "y_val",
                    "params:models",
                    "params:training"
                ],
                outputs=[
                    "trained_models",           # All trained models
                    "model_metrics",            # Performance metrics
                    "cv_metrics",               # Cross-validation metrics
                    "best_model_name",          # Name of best model
                    "best_model",               # Best model object
                    "best_model_metrics",       # Metrics of best model
                    "training_data_ref"         # Reference to training data
                ],
                name="train_models_with_cv",
                tags=["training", "cross_validation"],
            ),
            # Option 1: Use the main log_to_mlflow function
            node(
                func=log_to_mlflow,
                inputs=[
                    "best_model",               # best_model
                    "best_model_name",          # model_name
                    "best_model_metrics",       # metrics
                    "cv_metrics",               # cv_metrics
                    "training_data_ref",        # training_data
                    "params:models",            # model_params
                ],
                outputs="mlflow_run_id",
                name="log_best_model_to_mlflow",
                tags=["mlflow", "model_registry"],
            ),
        ]
    )