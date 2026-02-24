from fastapi.testclient import TestClient
from banking_customer_churn_prediction.api.main import app
import pytest
import numpy as np

client = TestClient(app)

class DummyModel:
    def predict_proba(self, X):
        # return 2D array with probabilities for two classes
        if hasattr(X, 'shape') and getattr(X, 'shape')[0] > 1:
            # batch
            return np.array([[0.8, 0.2]] * X.shape[0])
        return np.array([[0.2, 0.8]])

class DummyPreprocessor:
    def transform(self, df):
        # simply return a numpy array with correct number of rows
        try:
            n = df.shape[0]
        except Exception:
            n = 1
        return np.zeros((n, 5))

@pytest.fixture(autouse=True)
def mock_model_and_preprocessor(monkeypatch):
    import banking_customer_churn_prediction.api.dependencies as deps
    monkeypatch.setattr(deps, 'get_model', lambda: DummyModel())
    monkeypatch.setattr(deps, 'get_preprocessor', lambda: DummyPreprocessor())
    yield

def test_health_and_ready():
    r = client.get('/health')
    assert r.status_code == 200
    assert 'model_loaded' in r.json()

    r2 = client.get('/ready')
    assert r2.status_code == 200
    assert r2.json().get('ready') is True


def test_metrics():
    r = client.get('/metrics')
    assert r.status_code == 200
    assert 'help' in r.text or r.text.startswith('#')


def test_predict_single():
    payload = {
        "CustomerId": 123456,
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 1000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'churn_probability' in j
    assert 'risk_category' in j


def test_predict_batch():
    payload = {"customers": [
        {
            "CustomerId": 1,
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male",
            "Age": 40,
            "Tenure": 3,
            "Balance": 1000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0
        },
        {
            "CustomerId": 2,
            "CreditScore": 700,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 30,
            "Tenure": 2,
            "Balance": 2000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 60000.0
        }
    ]}
    r = client.post('/predict/batch', json=payload)
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list)
    assert len(arr) == 2
