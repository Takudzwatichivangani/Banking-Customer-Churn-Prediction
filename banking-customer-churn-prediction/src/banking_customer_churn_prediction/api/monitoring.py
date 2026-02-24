"""API monitoring and metrics."""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Callable

# Metrics
prediction_counter = Counter(
    'churn_predictions_total',
    'Total number of churn predictions',
    ['risk_category']
)

prediction_latency = Histogram(
    'churn_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

error_counter = Counter(
    'churn_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

active_connections = Gauge(
    'churn_api_active_connections',
    'Number of active connections'
)

model_performance = Gauge(
    'churn_model_performance',
    'Model performance metrics',
    ['metric']
)

def monitor_prediction(func):
    """Decorator to monitor prediction functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            prediction_latency.observe(latency)
            
            # Update risk category counter if available
            if hasattr(result, 'risk_category'):
                prediction_counter.labels(risk_category=result.risk_category).inc()
            else:
                prediction_counter.labels(risk_category='unknown').inc()
                
            return result
        except Exception as e:
            error_counter.labels(error_type=type(e).__name__).inc()
            raise
    return wrapper

def track_validation_error(error_type: str):
    """Track validation errors from FastAPI."""
    error_counter.labels(error_type=error_type).inc()


def monitor_request(func: Callable):
    """Decorator for FastAPI endpoints that catches validation errors."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_counter.labels(error_type=type(e).__name__).inc()
            raise
    return wrapper


def get_prediction_counts() -> dict:
    """Return a mapping of risk_category -> count for predictions.

    Uses the Prometheus metric `prediction_counter`'s collected samples in a
    safe way rather than relying on private attributes.
    """
    counts = {}
    for metric in prediction_counter.collect():
        for sample in metric.samples:
            labels = getattr(sample, 'labels', None) or {}
            val = getattr(sample, 'value', 0)
            rc = labels.get('risk_category') if isinstance(labels, dict) else None
            if rc:
                counts[rc] = counts.get(rc, 0) + val
    return counts


def get_total_predictions() -> int:
    """Return total number of predictions (sum of risk buckets)."""
    counts = get_prediction_counts()
    return int(sum(counts.values()))


def get_error_counts() -> dict:
    """Return a mapping of error_type -> count for prediction errors."""
    counts = {}
    for metric in error_counter.collect():
        for sample in metric.samples:
            labels = getattr(sample, 'labels', None) or {}
            val = getattr(sample, 'value', 0)
            et = labels.get('error_type') if isinstance(labels, dict) else None
            if et:
                counts[et] = counts.get(et, 0) + val
    return counts


def get_model_performance() -> dict:
    """Return model performance gauge values as a dict metric->value."""
    perf = {}
    for metric in model_performance.collect():
        for sample in metric.samples:
            labels = getattr(sample, 'labels', None) or {}
            val = getattr(sample, 'value', 0)
            m = labels.get('metric') if isinstance(labels, dict) else None
            if m:
                perf[m] = val
    return perf