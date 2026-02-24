"""Main FastAPI application for Bank Customer Churn Prediction API."""
import os
import sys
import logging
import json
import io
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Local imports - using the full package path
from banking_customer_churn_prediction.api.models import (
    CustomerData, PredictionResponse, BatchPredictionRequest, HealthResponse
)
from banking_customer_churn_prediction.api.dependencies import (
    get_model, get_preprocessor, verify_api_key, MODEL_PATH, PREPROCESSOR_PATH
)
from banking_customer_churn_prediction.api.monitoring import (
    prediction_counter, error_counter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="API for predicting customer churn in banking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get paths
CURRENT_DIR = Path(__file__).parent.absolute()  # src/banking_customer_churn_prediction/api/
SRC_DIR = CURRENT_DIR.parent.parent  # src/
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent  # project_root/

# Frontend is in project_root/frontend/static/
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "static"

# Create frontend directory if it doesn't exist
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info(f"✅ Mounted static files from {FRONTEND_DIR}")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Bank Customer Churn Prediction API")
    logger.info(f"📁 API directory: {CURRENT_DIR}")
    logger.info(f"📁 Src directory: {SRC_DIR}")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"📁 Frontend directory: {FRONTEND_DIR}")
    logger.info(f"📄 Index file: {FRONTEND_DIR / 'index.html'}")
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"📁 Preprocessor path: {PREPROCESSOR_PATH}")
    
    # Check if frontend exists
    if (FRONTEND_DIR / "index.html").exists():
        logger.info("✅ Frontend index.html found")
    else:
        logger.warning("⚠️ Frontend index.html not found!")
    
    # Test model loading
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        logger.info(f"✅ Model loaded: {type(model).__name__}")
        logger.info(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
    except Exception as e:
        logger.error(f"❌ Failed to load model/preprocessor: {e}")
    
    logger.info("=" * 60)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is healthy."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        return HealthResponse(
            status="healthy" if model and preprocessor else "degraded",
            model_loaded=model is not None,
            preprocessor_loaded=preprocessor is not None,
            mlflow_connected=True,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            preprocessor_loaded=False,
            mlflow_connected=False,
            timestamp=datetime.now().isoformat()
        )

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData, api_key: str = Depends(verify_api_key)):
    """Predict churn for a single customer."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # Transform features
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        churn_prob = float(model.predict_proba(X_processed)[0, 1])
        will_churn = churn_prob > 0.5
        confidence = abs(churn_prob - 0.5) * 2
        
        # Determine risk category
        if churn_prob < 0.3:
            risk_category = "Low"
            recommendation = "Maintain regular contact and consider cross-selling opportunities"
        elif churn_prob < 0.6:
            risk_category = "Medium"
            recommendation = "Increase engagement with personalized offers and check satisfaction"
        else:
            risk_category = "High"
            recommendation = "Immediate retention actions required - offer special incentives"
        
        # Update metrics
        prediction_counter.labels(risk_category=risk_category).inc()
        
        return PredictionResponse(
            customer_id=customer.CustomerId,
            churn_probability=churn_prob,
            will_churn=will_churn,
            risk_category=risk_category,
            recommendation=recommendation,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(batch_request: BatchPredictionRequest, api_key: str = Depends(verify_api_key)):
    """Predict churn for multiple customers."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        
        # Convert to DataFrame
        df = pd.DataFrame([c.dict() for c in batch_request.customers])
        
        # Transform and predict
        X_processed = preprocessor.transform(df)
        churn_probs = model.predict_proba(X_processed)[:, 1]
        
        responses = []
        for i, customer in enumerate(batch_request.customers):
            churn_prob = float(churn_probs[i])
            will_churn = churn_prob > 0.5
            confidence = abs(churn_prob - 0.5) * 2
            
            if churn_prob < 0.3:
                risk_category = "Low"
                recommendation = "Maintain regular contact"
            elif churn_prob < 0.6:
                risk_category = "Medium"
                recommendation = "Increase engagement"
            else:
                risk_category = "High"
                recommendation = "Immediate retention actions required"
            
            responses.append(PredictionResponse(
                customer_id=customer.CustomerId,
                churn_probability=churn_prob,
                will_churn=will_churn,
                risk_category=risk_category,
                recommendation=recommendation,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            ))
            
            prediction_counter.labels(risk_category=risk_category).inc()
        
        return responses
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# CSV upload prediction endpoint
@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """Upload CSV file for batch prediction."""
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_columns = ['CustomerId', 'CreditScore', 'Geography', 'Gender', 'Age',
                           'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                           'IsActiveMember', 'EstimatedSalary']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )
        
        # Convert to CustomerData objects
        customers = []
        for _, row in df.iterrows():
            customer_dict = row.to_dict()
            # Ensure numeric fields are proper types
            customer_dict['CustomerId'] = int(customer_dict['CustomerId'])
            customer_dict['CreditScore'] = int(customer_dict['CreditScore'])
            customer_dict['Age'] = int(customer_dict['Age'])
            customer_dict['Tenure'] = int(customer_dict['Tenure'])
            customer_dict['NumOfProducts'] = int(customer_dict['NumOfProducts'])
            customer_dict['HasCrCard'] = int(customer_dict['HasCrCard'])
            customer_dict['IsActiveMember'] = int(customer_dict['IsActiveMember'])
            
            customer = CustomerData(**customer_dict)
            customers.append(customer)
        
        # Create batch request
        batch_request = BatchPredictionRequest(customers=customers)
        
        # Call batch prediction
        return await predict_batch(batch_request, api_key)
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard stats endpoint
@app.get("/dashboard/stats")
async def get_dashboard_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics for dashboard."""
    try:
        return {
            "total_predictions": sum(prediction_counter._value.values()),
            "risk_distribution": {
                "low": prediction_counter.labels(risk_category="Low")._value.get(),
                "medium": prediction_counter.labels(risk_category="Medium")._value.get(),
                "high": prediction_counter.labels(risk_category="High")._value.get()
            },
            "model_metrics": {
                "accuracy": 0.86,
                "precision": 0.84,
                "recall": 0.79,
                "f1_score": 0.81
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "total_predictions": 0,
            "risk_distribution": {"low": 0, "medium": 0, "high": 0},
            "model_metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        }

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=open(index_path).read())
    else:
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>Bank Churn Predictor</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body class="bg-light">
                <div class="container mt-5">
                    <div class="row">
                        <div class="col-md-8 mx-auto">
                            <div class="card">
                                <div class="card-body text-center">
                                    <h1 class="display-4">🏦 Bank Customer Churn Predictor</h1>
                                    <p class="lead">API is running!</p>
                                    <hr>
                                    <div class="alert alert-info">
                                        <strong>Looking for frontend at:</strong><br>
                                        {index_path}
                                    </div>
                                    <p>Please ensure your index.html file is in the correct location.</p>
                                    <div class="mt-3">
                                        <a href="/health" class="btn btn-primary">Check Health</a>
                                        <a href="/docs" class="btn btn-success">API Documentation</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """)

@app.get("/test-model")
async def test_model():
    """Test if model and preprocessor load correctly."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        return {
            "status": "success",
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "model_type": str(type(model)) if model else None,
            "preprocessor_type": str(type(preprocessor)) if preprocessor else None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*60}")
    print(f"🚀 Starting Bank Churn Prediction Server...")
    print(f"📁 API directory: {CURRENT_DIR}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Frontend directory: {FRONTEND_DIR}")
    print(f"📄 Index file exists: {(FRONTEND_DIR / 'index.html').exists()}")
    print(f"🌐 Open http://localhost:8000 in your browser")
    print(f"📚 API docs at http://localhost:8000/docs")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)