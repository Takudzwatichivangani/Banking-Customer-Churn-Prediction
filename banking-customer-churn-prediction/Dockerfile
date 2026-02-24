FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-pin.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements-pin.txt ] ; then pip install --no-cache-dir -r requirements-pin.txt ; else pip install --no-cache-dir -r requirements.txt ; fi

COPY src/ ./src/
COPY data/ ./data/

# Expose configurable API port
ENV API_PORT=8000

# Create a startup script
RUN echo '#!/usr/bin/env python\nimport os\nimport sys\nsys.path.insert(0, "/app/src")\nfrom banking_customer_churn_prediction.api.main import app\nimport uvicorn\nport = int(os.getenv("API_PORT", "8000"))\nuvicorn.run(app, host="0.0.0.0", port=port)\n' > /app/start.py && \
    chmod +x /app/start.py

EXPOSE ${API_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["python", "/app/start.py"]