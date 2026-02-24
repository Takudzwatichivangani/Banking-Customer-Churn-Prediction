# run_all.py - FIXED VERSION
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run_backend():
    """Run FastAPI backend on port 8001"""
    print("🚀 Starting FastAPI backend on port 8001...")
    os.chdir(PROJECT_ROOT)
    
    # Set environment variables
    os.environ["MODEL_PATH"] = str(PROJECT_ROOT / "data" / "06_models" / "best_model.pkl")
    os.environ["PREPROCESSOR_PATH"] = str(PROJECT_ROOT / "data" / "04_feature" / "preprocessor.pkl")
    os.environ["API_KEYS"] = "test123"
    
    # Import and run your FastAPI app
    sys.path.insert(0, str(PROJECT_ROOT))
    
    try:
        from banking_customer_churn_prediction.api.main import app
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    except ImportError as e:
        print(f"❌ Error importing backend: {e}")
        print("Trying alternative method...")
        subprocess.run([sys.executable, "run.py"], shell=True)

def run_frontend():
    """Run React frontend on port 3000"""
    frontend_dir = PROJECT_ROOT / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return
    
    print("🌐 Starting React frontend on port 3000...")
    
    # Check if package.json exists
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print(f"❌ package.json not found in {frontend_dir}")
        return
    
    # Read package.json to check available scripts
    import json
    with open(package_json, 'r') as f:
        package_data = json.load(f)
    
    scripts = package_data.get('scripts', {})
    
    # Check which script to use
    if 'start' in scripts:
        script_to_run = 'start'
    elif 'dev' in scripts:
        script_to_run = 'dev'
    else:
        print("❌ No 'start' or 'dev' script found in package.json")
        print(f"Available scripts: {list(scripts.keys())}")
        return
    
    print(f"📦 Using npm script: '{script_to_run}'")
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Install dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], shell=True, capture_output=True)
    
    # Start React dev server - FIXED: Using subprocess.Popen instead of subprocess.run
    print(f"▶️  Running: npm {script_to_run}")
    subprocess.Popen(["npm", "run", "dev"], shell=True)

def main():
    print("=" * 60)
    print("🏦 BANKING CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 60)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    print("✅ Backend starting... (waiting 5 seconds)")
    time.sleep(5)
    
    # Start frontend in main thread
    try:
        run_frontend()
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        print("💡 Try running frontend manually: cd frontend && npm start")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")