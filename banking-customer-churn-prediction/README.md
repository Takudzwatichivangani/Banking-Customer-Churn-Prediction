# Banking Customer Churn Prediction API

A production-ready FastAPI service for predicting customer churn in banking. Built with scikit-learn, Prometheus monitoring, and full containerization support.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Deployment](#deployment)
- [Production Hardening](#production-hardening)
- [CI/CD](#cicd)
- [Troubleshooting](#troubleshooting)

## Features

✅ **FastAPI + Uvicorn** — High-performance REST API with auto-generated API docs  
✅ **Model Management** — Load models from MLflow, local storage, or S3 with fallback  
✅ **Readiness & Health Checks** — `/ready` and `/health` endpoints for orchestration  
✅ **Prometheus Metrics** — `/metrics` endpoint for monitoring predictions and errors  
✅ **JWT/API Key Auth** — Header-based API key authentication (disabled in dev, required in production)  
✅ **Batch Prediction** — Single and batch prediction endpoints; CSV upload support  
✅ **Docker & Compose** — Multi-stage Dockerfile; dev stack with monitoring (Prometheus, Grafana)  
✅ **Unit Tests** — 8 passing tests covering core endpoints  
✅ **GitHub Actions CI** — Automated testing and optional image build/push to GHCR  
✅ **Pinned Dependencies** — `requirements-pin.txt` for reproducible builds  

## Project Structure

```
banking-customer-churn-prediction/
├── src/banking_customer_churn_prediction/
│   ├── api/
│   │   ├── main.py              # FastAPI app, endpoints, startup
│   │   ├── dependencies.py      # Model/preprocessor loading, API key auth, S3 fallback
│   │   ├── monitoring.py        # Prometheus metrics
│   │   └── models.py            # Pydantic request/response schemas
│   └── pipelines/               # Kedro pipelines (data processing, training, inference)
├── tests/
│   ├── test_api.py              # Unit tests for endpoints
│   └── conftest.py              # Pytest fixtures
├── data/
│   ├── 04_feature/              # Fitted preprocessor
│   └── 06_models/               # Trained model
├── monitoring/
│   ├── prometheus/              # Prometheus config
│   └── grafana/                 # Grafana dashboards
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI/CD
├── Dockerfile                   # Production image
├── docker-compose.dev.yml       # Local dev stack
├── requirements.txt             # Main dependencies
├── requirements-dev.txt         # Dev dependencies (pytest, flake8)
├── requirements-pin.txt         # Pinned dependencies (reproducible builds)
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)

### Local Development

1. **Clone and install dependencies:**
```bash
git clone https://github.com/Takudzwatichivangani/Banking-Customer-Churn-Prediction.git
cd Banking-Customer-Churn-Prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Run tests:**
```bash
pytest -q
```

3. **Start the API locally (development mode):**
```bash
python -m uvicorn banking_customer_churn_prediction.api.main:app --reload --port 8000
```

4. **Access API:**
   - Docs: http://localhost:8000/docs
   - API: http://localhost:8000

### Local Development with Docker Compose

Bring up API + Prometheus + Grafana stack:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

- API: http://localhost:8001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Stop services:
```bash
docker compose -f docker-compose.dev.yml down
```

## API Endpoints

### Health & Readiness

- **GET `/health`** — Liveness check; returns model/preprocessor status
- **GET `/ready`** — Readiness probe; returns 200 only when artifacts are loaded

### Prediction

- **POST `/predict`** — Single customer prediction
  - Request: `CustomerData` (customer attributes)
  - Response: `PredictionResponse` (churn probability, risk category, recommendation)

- **POST `/predict/batch`** — Batch prediction for multiple customers
  - Request: `BatchPredictionRequest` (list of customers)
  - Response: List of `PredictionResponse`

- **POST `/predict/upload`** — Upload CSV for batch prediction
  - File: CSV with required customer columns
  - Response: List of predictions

### Monitoring

- **GET `/metrics`** — Prometheus metrics (predictions, errors, model performance)
- **GET `/dashboard/stats`** — Dashboard statistics (total predictions, risk distribution)

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test123" \
  -d '{
    "CustomerId": 15634602,
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

## Environment Variables

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment mode (`development` or `production`) |
| `API_PORT` | `8000` | Port the API listens on |
| `API_KEYS` | `test123,development` | Comma-separated list of valid API keys (required in production) |

### Model & Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./data/06_models/best_model.pkl` | Path to trained model |
| `PREPROCESSOR_PATH` | `./data/04_feature/preprocessor.pkl` | Path to fitted preprocessor |
| `METRICS_PATH` | `./data/06_models/model_metrics.json` | Path to model metrics |
| `MLFLOW_MODEL_URI` | — | MLflow model URI (optional; overrides local model) |
| `MODEL_S3_URI` | — | S3 URI (`s3://bucket/key`) for model fallback |
| `AUTO_FIT_SAVE` | `false` | Allow auto-fitted preprocessor to be saved (dev only) |

### Observability (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTRY_DSN` | — | Sentry error reporting (optionally activated if set) |
| `ENABLE_RATE_LIMITING` | `false` | Enable slowapi rate limiting |

### Deployment & Secrets

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | — | AWS region for S3 access |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials (prefer IAM role in production) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials (prefer IAM role in production) |

### Example `.env` File

```
ENV=development
API_PORT=8000
API_KEYS=test123,development
AUTO_FIT_SAVE=false
MODEL_PATH=./data/06_models/best_model.pkl
PREPROCESSOR_PATH=./data/04_feature/preprocessor.pkl
```

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v
pytest -q  # Brief output
pytest tests/test_api.py::test_predict_single  # Single test
```

### Code Style & Linting

```bash
flake8 src/
```

### Building the Docker Image

```bash
docker build -t churn-api:local .
```

### Running a Container Locally

```bash
docker run --rm -p 8000:8000 \
  -e API_PORT=8000 \
  -e ENV=development \
  churn-api:local
```

With mounted data:
```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e API_PORT=8000 \
  churn-api:local
```

## Deployment

### Production Readiness Checklist

- [ ] Set `ENV=production` in deployment configuration
- [ ] Provision `API_KEYS` (or integrate with an identity provider)
- [ ] Move secrets to a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
- [ ] Configure S3 IAM role for model downloads (avoid storing credentials in env vars)
- [ ] Deploy to production orchestration (ECS Fargate, Kubernetes, etc.)
- [ ] Configure health checks on `/ready` endpoint (orchestration)
- [ ] Set up monitoring (Prometheus scrape, Grafana dashboards, Sentry)
- [ ] Enable rate limiting with `ENABLE_RATE_LIMITING=true` (optional, requires slowapi installed)

### Deploy to Docker Hub / GHCR

The GitHub Actions workflow will automatically build and push images when you push to `main`:

1. **GHCR (GitHub Container Registry)** — Uses `GITHUB_TOKEN` (default, no secret setup needed)
2. **Docker Hub** — Requires repo secrets:
   - `DOCKER_REGISTRY` (e.g., `docker.io`)
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

Set Docker Hub secrets in GitHub repo **Settings → Secrets → Actions** if desired.

### Kubernetes Deployment (Example)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: churn-api
spec:
  containers:
  - name: api
    image: ghcr.io/takudzwatichivangani/bank-churn-api:latest
    ports:
    - containerPort: 8000
    env:
    - name: ENV
      value: "production"
    - name: API_PORT
      value: "8000"
    - name: API_KEYS
      valueFrom:
        secretKeyRef:
          name: api-secrets
          key: api-keys
    - name: MODEL_S3_URI
      value: "s3://my-models/churn-model.pkl"
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
```

### AWS ECS Fargate Deployment

1. Push image to ECR
2. Create ECS task definition with:
   - Image URI pointing to ECR image
   - Container port 8000
   - Environment variables (API_PORT, ENV, API_KEYS from Secrets Manager)
   - IAM task role for S3 access
3. Create ECS service with ALB health checks on `/ready`

## Production Hardening

### Authentication

- In **development** (`ENV=development`): API key header optional
- In **production** (`ENV=production`): API key header required; requests without valid `X-API-Key` return 401

### Preprocessor Auto-Fit

- By default, auto-fitted preprocessors are NOT saved to disk
- Enable saving only in dev with `AUTO_FIT_SAVE=true`
- In production, fail-fast if preprocessor is unfitted

### Model Fallback Chain

1. Load model from `MLFLOW_MODEL_URI` if set
2. Load from local `MODEL_PATH`
3. If missing and `MODEL_S3_URI` set, download from S3 with 3 retry attempts
4. If all fail, crash early (prevents silent degradation)

### Secrets Management

- **Never** hardcode secrets in code or images
- Use IAM roles (AWS), Kubernetes secrets, or HashiCorp Vault
- Rotate credentials regularly
- Use environment variables for non-sensitive config

## CI/CD

### GitHub Actions Workflow

Triggered on push/PR to `main`:

1. **Test Job** — Runs pytest, linting
2. **Build & Push Job** — (if tests pass and on main branch)
   - Builds Docker image using `requirements-pin.txt`
   - Pushes to GHCR with tags `:latest` and `:sha`

### Running Tests Locally

```bash
pytest -q
```

### Enabling Image Scan in CI

Add to `.github/workflows/ci.yml` if using Docker Scout:

```yaml
- name: Scan image
  uses: docker/scout-action@v1
  with:
    image: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}
```

## Troubleshooting

### API fails to start: "Preprocessor not fitted in production"

**Cause:** Preprocessor file is missing or unfitted.

**Fix:**
- Ensure `PREPROCESSOR_PATH` points to a valid, fitted preprocessor
- In dev, set `AUTO_FIT_SAVE=true` to auto-fit and save
- In production, provide a properly fitted preprocessor

### S3 download fails: "boto3 is required"

**Cause:** `MODEL_S3_URI` is set but `boto3` not installed.

**Fix:**
```bash
pip install boto3
# or
pip install -r requirements-pin.txt
```

### "Invalid API key" in production

**Cause:** Request missing `X-API-Key` header or key not in `API_KEYS`.

**Fix:**
- Add header: `curl -H "X-API-Key: your-key" ...`
- Ensure key is in `API_KEYS` env var (comma-separated)
- In dev, set `ENV=development` to skip key validation

### Tests fail locally but pass in CI

**Cause:** Dependency version mismatch.

**Fix:**
```bash
pip install -r requirements-pin.txt  # Use pinned versions
pip uninstall -r requirements.txt && pip install -r requirements-pin.txt
```

### Docker image build fails

**Cause:** Missing data files or permissions.

**Fix:**
```bash
# Ensure data directory exists
mkdir -p data/{04_feature,06_models}

# Rebuild
docker build --no-cache -t churn-api:local .
```

### Prometheus scrape fails

**Cause:** Container not exposing metrics or incorrect scrape config.

**Fix:**
- Ensure `docker-compose.dev.yml` mounts correct prometheus config
- Check `monitoring/prometheus/prometheus.yml` has correct target
- Verify container is running: `docker compose ps`

## Support & Contributing

For issues, feature requests, or contributions, open an issue or PR on [GitHub](https://github.com/Takudzwatichivangani/Banking-Customer-Churn-Prediction).

## License

[Insert your license here]
