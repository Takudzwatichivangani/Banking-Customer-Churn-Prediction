Production README — Banking Customer Churn API

Summary
- Multi-stage Docker image; app served by Uvicorn + FastAPI.
- Exposes /health (liveness) and /ready (readiness — checks model artifacts).
- Model artifacts expected at /app/data/04_models/best_model.pkl and /app/data/04_feature/preprocessor_fitted.pkl by default.

1) Create a reproducible pinned requirements file
- Install pip-tools in your venv:
  python -m pip install pip-tools
- Generate pinned file (with hashes):
  pip-compile --generate-hashes --output-file=requirements-pin.txt requirements.in

2) Local test & build
- Install deps for local testing:
  python -m pip install -r requirements-pin.txt
  python -m pip install -e .
- Run tests:
  pytest -q
- Build image:
  docker build --no-cache -t churn-api:prod .
- Run container (bind host port 8000):
  docker run --rm -p 8000:8000 churn-api:prod

Dev option (mount local data instead of baking):
  docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" churn-api:prod

3) Production practices (recommended)
- Store model artifacts in S3 (versioned). Do not commit production models to Git.
- In CI, fetch a specific model version from artifact store and either:
  - Bake into the image (not recommended for frequent model updates), or
  - Upload images without artifacts and let the container download the model at startup (preferred).
- Use ECR + ECS Fargate for deployment:
  - CI builds & pushes image to ECR.
  - ECS task uses an IAM role to fetch models from S3 and write to /tmp/models (local cache).
  - ALB health checks on /ready (should return 200 only when artifacts are present).
- Use Secrets Manager / SSM for any credentials; do not store creds in the image.

4) Environment variables
- MODEL_DIR: path inside container for artifacts (default /app/data). Use S3 URI if implementing S3 fetch.
- AWS_* credentials: use IAM role attached to ECS task instead of env vars where possible.

New environment variables introduced by recent updates
- `API_PORT`: port the API listens on (default `8000`).
- `AUTO_FIT_SAVE`: when `true`, allows auto-fitted preprocessors to be saved back to `PREPROCESSOR_PATH` (development only).
- `API_KEYS`: comma-separated list of valid API keys for `X-API-Key` header (default `test123,development`).
- `MLFLOW_MODEL_URI`: MLflow model URI to load a model from MLflow.
- `MODEL_S3_URI`: optional `s3://bucket/key` to download the model at startup if local model is missing.

Notes on S3/model fetching
- If the local model file is missing, the API will attempt to download from `MODEL_S3_URI` (requires `boto3`). The downloader retries a few times before failing.
- For production use prefer assigning an IAM role to the runtime environment instead of storing AWS credentials in env vars.

5) CI suggestion
- CI steps: lint → tests → build image → scan image → push to ECR → trigger deployment.
- Add integration test job that runs the container with mounted test artifacts and calls /predict.

6) Observability
- Add structured logging, metrics endpoint (Prometheus), and error reporting (Sentry) for production.

Running tests locally
- Install dev deps: `pip install -r requirements-dev.txt`
- Run tests: `pytest -q`

Local dev compose stack
- Use `docker-compose -f docker-compose.dev.yml up --build` to start the API with Prometheus and Grafana for local development.