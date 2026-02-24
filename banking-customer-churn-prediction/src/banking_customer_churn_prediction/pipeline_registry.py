"""Pipeline registry for the churn prediction project."""
from typing import Dict
from kedro.pipeline import Pipeline

from banking_customer_churn_prediction.pipelines.etl.pipeline import create_pipeline as create_etl_pipeline
from banking_customer_churn_prediction.pipelines.feature_engineering.pipeline import create_pipeline as create_feature_engineering_pipeline
from banking_customer_churn_prediction.pipelines.preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline
from banking_customer_churn_prediction.pipelines.training.pipeline import create_pipeline as create_training_pipeline
from banking_customer_churn_prediction.pipelines.inference.pipeline import create_pipeline as create_inference_pipeline
from banking_customer_churn_prediction.pipelines.monitoring.pipeline import create_pipeline as create_monitoring_pipeline
from banking_customer_churn_prediction.pipelines.postprocessing.pipeline import create_pipeline as create_postprocessing_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    
    # Individual pipelines
    etl_pipeline = create_etl_pipeline()
    feature_engineering_pipeline = create_feature_engineering_pipeline()
    preprocessing_pipeline = create_preprocessing_pipeline()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    monitoring_pipeline = create_monitoring_pipeline()
    postprocessing_pipeline = create_postprocessing_pipeline()
    
    # Master pipelines
    data_preparation_pipeline = (
        etl_pipeline
        + feature_engineering_pipeline
        + preprocessing_pipeline
    )
    
    model_development_pipeline = (
        data_preparation_pipeline
        + training_pipeline
    )
    
    batch_prediction_pipeline = (
        data_preparation_pipeline
        + inference_pipeline
        + postprocessing_pipeline
    )
    
    full_pipeline = (
        etl_pipeline
        + feature_engineering_pipeline
        + preprocessing_pipeline
        + training_pipeline
        + inference_pipeline
        + monitoring_pipeline
        + postprocessing_pipeline
    )
    
    return {
        # Individual pipelines (for debugging)
        "etl": etl_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "preprocessing": preprocessing_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        "monitoring": monitoring_pipeline,
        "postprocessing": postprocessing_pipeline,
        
        # Combined pipelines (for workflows)
        "data_preparation": data_preparation_pipeline,
        "model_development": model_development_pipeline,
        "batch_prediction": batch_prediction_pipeline,
        
        # Full pipeline
        "__default__": model_development_pipeline,  # Default runs model development
        "full": full_pipeline,
    }