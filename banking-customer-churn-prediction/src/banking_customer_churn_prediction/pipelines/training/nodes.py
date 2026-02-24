"""Model training nodes with cross-validation and MLflow integration."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# OPTIONAL XGBOOST IMPORT - Safe for API mode
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    # Create a placeholder for API mode (when xgboost is not installed)
    class XGBClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "xgboost is not installed. This is expected in API serving mode. "
                "Install xgboost only for training environments: pip install xgboost"
            )
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)

logger = logging.getLogger(__name__)

def train_models_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    model_params: Dict[str, Dict],
    training_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Train multiple models with cross-validation and evaluate on validation set.
    Returns trained_models, model_metrics, cv_metrics, and best model info.
    """
    logger.info(f"Training models on {X_train.shape[0]} samples")
    
    results = {}
    trained_models = {}
    cv_metrics = {}
    
    # Initialize CV strategy
    cv = StratifiedKFold(
        n_splits=training_params.get("cv_folds", 5),
        shuffle=True,
        random_state=model_params.get("logistic_regression", {}).get("random_state", 42)
    )
    
    # Model configurations - DYNAMIC based on availability
    model_configs = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": model_params.get("logistic_regression", {})
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": model_params.get("random_forest", {})
        }
    }
    
    # Only add xgboost if it's available AND configured
    if XGBOOST_AVAILABLE and "xgboost" in model_params:
        model_configs["xgboost"] = {
            "class": XGBClassifier,
            "params": model_params.get("xgboost", {})
        }
    elif not XGBOOST_AVAILABLE:
        logger.info("xgboost not available - skipping xgboost model training")
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # Initialize model
            model_class = config["class"]
            params = config["params"]
            model = model_class(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring=training_params.get("scoring_metric", "roc_auc"),
                n_jobs=-1
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Validation predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "roc_auc": roc_auc_score(y_val, y_val_proba),
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred),
                "recall": recall_score(y_val, y_val_pred),
                "f1": f1_score(y_val, y_val_pred),
                "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist()
            }
            
            # Store results
            trained_models[model_name] = model
            cv_metrics[model_name] = {
                "scores": cv_scores.tolist(),
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std())
            }
            results[model_name] = metrics
            
            logger.info(f"{model_name} - CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"{model_name} - Val AUC: {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Select best model
    if results:
        best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
        best_model = trained_models[best_model_name]
        best_metrics = results[best_model_name]
        
        # Also return X_train and X_val for MLflow signature
        training_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val
        }
        
        logger.info(f"Best model: {best_model_name} with AUC: {best_metrics['roc_auc']:.4f}")
        
        return trained_models, results, cv_metrics, best_model_name, best_model, best_metrics, training_data
    else:
        raise ValueError("No models were successfully trained")

def log_to_mlflow(model, model_name, metrics, cv_metrics, training_data_ref, models_config):
    """Log model to MLflow with better error handling."""
    import mlflow
    from mlflow.exceptions import MlflowException
    import datetime
    
    try:
        # Try different experiment names
        experiment_names = [
            f"churn_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "churn_prediction_latest",
            "customer_churn"
        ]
        
        for exp_name in experiment_names:
            try:
                mlflow.set_experiment(exp_name)
                break
            except MlflowException:
                continue
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log parameters
            model_params = models_config.get(model_name, {})
            if model_params:
                mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics({
                "val_auc": metrics.get("validation_auc", 0),
                "cv_mean_auc": cv_metrics.get("mean", 0),
                "cv_std_auc": cv_metrics.get("std", 0)
            })
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"churn_model_{model_name}"
            )
            
            # Log training data reference
            if training_data_ref is not None:
                mlflow.log_dict(training_data_ref, "training_data_ref.json")
            
            print(f"Logged to MLflow run: {run_id}")
            return run_id  # ✅ Return a value
            
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print("Continuing without MLflow...")
        # Return a placeholder value instead of None
        return "mlflow_failed_but_model_saved"  # ✅ Return a string, not None

def log_model_simple(
    best_model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    model_params: Dict[str, Dict]
) -> str:
    """
    Simplified MLflow logging without training data.
    Use this if the main log_to_mlflow function has issues.
    """
    mlflow.set_experiment("churn_prediction")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log parameters
        best_model_params = model_params.get(model_name, {})
        for param_name, param_value in best_model_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=f"churn_model_{model_name}"
        )
        
        logger.info(f"Logged model (simple) to MLflow run: {run_id}")
        
        return run_id