# run_production.py
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Set environment variables for production
    os.environ["ENV"] = "production"
    os.environ["MODEL_PATH"] = str(Path("data/06_models/best_model.pkl").absolute())
    os.environ["PREPROCESSOR_PATH"] = str(Path("data/04_feature/preprocessor.pkl").absolute())
    # Allow overriding the API port via env var
    os.environ.setdefault("API_PORT", "8000")
    
    print("="*60)
    print("🚀 Starting Production Bank Churn Prediction API")
    print("📁 Model path:", os.environ["MODEL_PATH"])
    print("📁 Preprocessor path:", os.environ["PREPROCESSOR_PATH"])
    print("📊 Monitoring: http://localhost:3000 (Grafana)")
    print("📈 Metrics: http://localhost:9090 (Prometheus)")
    print("🔔 Alerts: http://localhost:9093 (Alertmanager)")
    print("="*60)
    
    # Run with production settings
    api_port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "src.banking_customer_churn_prediction.api.main:app",
        host="0.0.0.0",
        port=api_port,
        workers=4,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*"
    )