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
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

# Local imports - using the full package path
from banking_customer_churn_prediction.api.models import (
    CustomerData, PredictionResponse, BatchPredictionRequest, HealthResponse
)
from banking_customer_churn_prediction.api.dependencies import (
    get_model, get_preprocessor, verify_api_key_header, MODEL_PATH, PREPROCESSOR_PATH
)
from banking_customer_churn_prediction.api.monitoring import (
    prediction_counter, error_counter, get_prediction_counts, get_total_predictions, get_model_performance
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
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent  # banking-customer-churn-prediction/
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "static"

# Create frontend directory if it doesn't exist
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info(f"✅ Mounted static files from {FRONTEND_DIR}")

# Custom dependency to handle API key in multiple formats
async def verify_api_key_dependency(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key_header)
) -> str:
    """Verify API key from header, query parameter, or Authorization header.

    This wrapper tries multiple places for a key but relies on
    `verify_api_key_header` for validation logic.
    """
    # If the dependency resolved a key, return it
    if api_key:
        return api_key
    # In non-production allow missing API key for development/testing
    env = os.getenv("ENV", "development")
    if env != "production":
        logger.debug("No API key provided but ENV != production — allowing access for development/testing")
        return None

    # Try to get from query parameters
    query_api_key = request.query_params.get('api_key')
    if query_api_key:
        return query_api_key

    # Try to get from Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.replace('Bearer ', '')

    logger.warning(f"No API key found. Headers: {dict(request.headers)}")
    raise HTTPException(status_code=401, detail="API key is required")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Bank Customer Churn Prediction API")
    logger.info(f"📁 API directory: {CURRENT_DIR}")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"📁 Frontend directory: {FRONTEND_DIR}")
    logger.info(f"📄 Index file: {FRONTEND_DIR / 'index.html'}")
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"📁 Preprocessor path: {PREPROCESSOR_PATH}")
    
    # Check if frontend exists
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        logger.info(f"✅ Frontend index.html found at {index_file}")
    else:
        logger.warning(f"⚠️ Frontend index.html not found at {index_file}")
    
    # Test model loading
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        logger.info(f"✅ Model loaded: {type(model).__name__}")
        logger.info(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
    except Exception as e:
        logger.error(f"❌ Failed to load model/preprocessor: {e}")
    
    logger.info("=" * 60)


@app.get("/ready")
async def readiness_check():
    """Readiness endpoint: returns 200 only when model and preprocessor are available."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        ready = bool(model is not None and preprocessor is not None)
        status = 200 if ready else 503
        return JSONResponse(status_code=status, content={
            "ready": ready,
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(status_code=503, content={
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for scraping."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

# Test endpoint
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working."""
    return {"status": "ok", "message": "API is working correctly"}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(api_key: str = Depends(verify_api_key_dependency)):
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
async def predict(
    customer: CustomerData, 
    api_key: str = Depends(verify_api_key_dependency)
):
    """Predict churn for a single customer."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        
        # Use model_dump() for Pydantic V2
        customer_dict = customer.model_dump()
        logger.info(f"Received customer data: {customer_dict}")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_dict])
        
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
async def predict_batch(
    batch_request: BatchPredictionRequest, 
    api_key: str = Depends(verify_api_key_dependency)
):
    """Predict churn for multiple customers."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        
        # Use model_dump() for each customer
        customers_data = [c.model_dump() for c in batch_request.customers]
        logger.info(f"Received batch with {len(customers_data)} customers")
        
        # Convert to DataFrame
        df = pd.DataFrame(customers_data)
        
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
async def predict_upload(
    file: UploadFile = File(...), 
    api_key: str = Depends(verify_api_key_dependency)
):
    """Upload CSV file for batch prediction."""
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"CSV shape: {df.shape}")
        
        # Validate columns
        required_columns = ['CustomerId', 'CreditScore', 'Geography', 'Gender', 'Age',
                           'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                           'IsActiveMember', 'EstimatedSalary']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            error_msg = f"Missing columns: {missing_columns}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert to CustomerData objects
        customers = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                customer_dict = {}
                for col in required_columns:
                    value = row[col]
                    
                    # Handle NaN values
                    if pd.isna(value):
                        if col in ['Balance', 'EstimatedSalary']:
                            value = 0.0
                        elif col in ['CreditScore', 'Age', 'Tenure', 'NumOfProducts']:
                            value = 0
                        else:
                            value = '' if col in ['Geography', 'Gender'] else 0
                    
                    # Convert types
                    if col in ['CustomerId', 'CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']:
                        try:
                            customer_dict[col] = int(float(value))
                        except:
                            customer_dict[col] = 0
                    elif col in ['Balance', 'EstimatedSalary']:
                        try:
                            customer_dict[col] = float(value)
                        except:
                            customer_dict[col] = 0.0
                    else:
                        customer_dict[col] = str(value)
                
                customer = CustomerData(**customer_dict)
                customers.append(customer)
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        if errors:
            logger.warning(f"Some rows had errors: {errors[:5]}")
        
        if not customers:
            raise HTTPException(status_code=400, detail="No valid customers found in CSV")
        
        logger.info(f"Successfully parsed {len(customers)} customers")
        
        # Create batch request
        batch_request = BatchPredictionRequest(customers=customers)
        
        # Call batch prediction
        return await predict_batch(batch_request, api_key)
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard stats endpoint
@app.get("/dashboard/stats")
async def get_dashboard_stats(api_key: str = Depends(verify_api_key_dependency)):
    """Get statistics for dashboard using safe metric accessors."""
    try:
        counts = get_prediction_counts()
        perf = get_model_performance()

        return {
            "total_predictions": get_total_predictions(),
            "risk_distribution": {
                "low": int(counts.get('Low', 0)),
                "medium": int(counts.get('Medium', 0)),
                "high": int(counts.get('High', 0))
            },
            "model_metrics": perf or {}
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "total_predictions": 0,
            "risk_distribution": {"low": 0, "medium": 0, "high": 0},
            "model_metrics": {}
        }

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"✅ Successfully loaded frontend from {index_path}")
            return HTMLResponse(content=content)
        except Exception as e:
            logger.error(f"Failed to read index.html: {e}")
    
    # Fallback HTML
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head><title>Bank Churn API</title></head>
    <body>
        <h1>Bank Customer Churn Prediction API</h1>
        <p>API is running! Frontend file not found at: {index_path}</p>
        <p><a href="/docs">API Documentation</a></p>
        <p><a href="/test">Test Endpoint</a></p>
    </body>
    </html>
    """)

@app.get("/test-paths")
async def test_paths(api_key: str = Depends(verify_api_key_dependency)):
    """Test endpoint to verify all paths."""
    return {
        "current_dir": str(CURRENT_DIR),
        "project_root": str(PROJECT_ROOT),
        "frontend_dir": str(FRONTEND_DIR),
        "index_file_exists": (FRONTEND_DIR / "index.html").exists(),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
        "model_path_exists": Path(MODEL_PATH).exists() if MODEL_PATH else False,
        "preprocessor_path_exists": Path(PREPROCESSOR_PATH).exists() if PREPROCESSOR_PATH else False
    }

if __name__ == "__main__":
    import uvicorn
    api_port = int(os.getenv("API_PORT", "8000"))
    print(f"\n{'='*60}")
    print(f"🚀 Starting Bank Churn Prediction Server...")
    print(f"📁 API directory: {CURRENT_DIR}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Frontend directory: {FRONTEND_DIR}")
    print(f"📄 Index file exists: {(FRONTEND_DIR / 'index.html').exists()}")
    print(f"🌐 Open http://localhost:{api_port} in your browser")
    print(f"🔍 Test endpoint: http://localhost:{api_port}/test")
    print(f"📚 API docs: http://localhost:{api_port}/docs")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="127.0.0.1", port=api_port)