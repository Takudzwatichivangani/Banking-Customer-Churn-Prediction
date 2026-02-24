"""Pydantic models for the API."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np

class CustomerData(BaseModel):
    """Input data model for customer prediction."""
    CustomerId: int = Field(..., description="Customer ID", example=15634602)
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score", example=619)
    Geography: str = Field(..., description="Country", example="France")
    Gender: str = Field(..., description="Gender", example="Male")
    Age: int = Field(..., ge=18, le=100, description="Age", example=42)
    Tenure: int = Field(..., ge=0, le=10, description="Tenure in years", example=2)
    Balance: float = Field(..., ge=0, description="Account balance", example=0.0)
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products", example=1)
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card", example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member", example=1)
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary", example=101348.88)
    
    @validator('Geography')
    def validate_geography(cls, v):
        valid = ['France', 'Spain', 'Germany']
        if v not in valid:
            raise ValueError(f'Geography must be one of {valid}')
        return v
    
    @validator('Gender')
    def validate_gender(cls, v):
        valid = ['Male', 'Female']
        if v not in valid:
            raise ValueError(f'Gender must be one of {valid}')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    customer_id: int = Field(..., description="Customer ID")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability")
    will_churn: bool = Field(..., description="Will churn prediction")
    risk_category: str = Field(..., description="Risk category")
    recommendation: str = Field(..., description="Business recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    customers: List[CustomerData] = Field(..., min_items=1, max_items=1000)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loaded status")
    preprocessor_loaded: bool = Field(..., description="Preprocessor loaded status")
    mlflow_connected: bool = Field(..., description="MLflow connection status")
    timestamp: str = Field(..., description="Check timestamp")

class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str = Field(..., description="Model type")
    model_version: str = Field(..., description="Model version")
    num_features: int = Field(..., description="Number of features")
    features: List[str] = Field(..., description="Feature names")
    training_date: str = Field(..., description="Training date")
    performance: Dict[str, float] = Field(..., description="Performance metrics")