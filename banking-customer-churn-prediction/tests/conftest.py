"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_customer_data():
    """Complete sample customer data - 10 samples for proper testing."""
    return pd.DataFrame({
        'RowNumber': list(range(1, 11)),
        'CustomerId': list(range(15634602, 15634612)),
        'Surname': ['Hargrave', 'Hill', 'Onio', 'Boni', 'Mitchell', 
                   'Cooper', 'Lee', 'Scott', 'King', 'Wright'],
        'CreditScore': [619, 608, 502, 699, 850, 645, 577, 603, 710, 680],
        'Geography': ['France', 'Spain', 'France', 'France', 'Spain', 
                     'Germany', 'France', 'Spain', 'Germany', 'France'],
        'Gender': ['Female', 'Female', 'Female', 'Female', 'Female',
                  'Male', 'Male', 'Male', 'Female', 'Male'],
        'Age': [42, 41, 42, 39, 43, 44, 29, 38, 31, 37],
        'Tenure': [2, 1, 8, 1, 2, 8, 4, 3, 2, 6],
        'Balance': [0.0, 83807.86, 159660.8, 0.0, 125510.82,
                   113755.78, 0.0, 149756.71, 115047.34, 102016.72],
        'NumOfProducts': [1, 1, 3, 2, 1, 2, 1, 2, 2, 1],
        'HasCrCard': [1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        'IsActiveMember': [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.1,
                          149756.71, 10062.8, 119346.88, 129188.45, 108418.89],
        'Exited': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # 4 churned, 6 not churned
    })

@pytest.fixture
def sample_features():
    """Sample features after preprocessing."""
    return pd.DataFrame({
        'CreditScore': [619, 608, 502, 699, 850],
        'Age': [42, 41, 42, 39, 43],
        'Tenure': [2, 1, 8, 1, 2],
        'Balance': [0.0, 83807.86, 159660.8, 0.0, 125510.82],
        'NumOfProducts': [1, 1, 3, 2, 1],
        'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.1],
        'avg_bal_per_product': [0.0, 83807.86, 53220.27, 0.0, 125510.82],
        'tenure_bucket_0-2': [1, 1, 0, 1, 1],
        'tenure_bucket_3-5': [0, 0, 0, 0, 0],
        'tenure_bucket_6-10': [0, 0, 1, 0, 0],
        'Geography_France': [1, 0, 1, 1, 0],
        'Geography_Spain': [0, 1, 0, 0, 1],
        'Geography_Germany': [0, 0, 0, 0, 0],
        'Gender_Female': [1, 1, 1, 1, 1],
        'Gender_Male': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def sample_target():
    """Sample target variable."""
    return pd.Series([1, 0, 1, 0, 0], name='Exited')

@pytest.fixture
def mock_model():
    """Mock sklearn model for testing."""
    class MockModel:
        def __init__(self):
            self.predict_calls = []
            self.predict_proba_calls = []
        
        def predict(self, X):
            self.predict_calls.append(X)
            return np.array([0, 1, 0, 1, 0])
        
        def predict_proba(self, X):
            self.predict_proba_calls.append(X)
            return np.array([
                [0.8, 0.2],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.85, 0.15]
            ])
    
    return MockModel()

@pytest.fixture
def test_parameters():
    """Test parameters matching your project."""
    return {
        'data': {
            'target': 'Exited',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'stratify': True,
            'columns': {
                'numeric': ['CreditScore', 'Age', 'Tenure', 'Balance', 
                           'NumOfProducts', 'EstimatedSalary'],
                'categorical': ['Geography', 'Gender'],
                'drop': ['RowNumber', 'CustomerId', 'Surname']
            }
        },
        'features': {
            'derived': [
                {'name': 'avg_bal_per_product', 'formula': 'Balance / (NumOfProducts + 1e-9)'},
                {'name': 'tenure_bucket', 'type': 'categorical', 'bins': [-1, 2, 5, 10, 50]}
            ],
            'preprocessing': {
                'numeric': {'impute_strategy': 'median', 'scale_strategy': 'standard'},
                'categorical': {'impute_strategy': 'most_frequent', 'encode_strategy': 'onehot'}
            }
        },
        'models': {
            'logistic_regression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
        },
        'training': {
            'cv_folds': 5,
            'scoring_metric': 'roc_auc',
            'verbose': 1
        }
    }

@pytest.fixture
def small_sample_data():
    """Smaller sample for quick tests."""
    return pd.DataFrame({
        'RowNumber': [1, 2],
        'CustomerId': [1, 2],
        'Surname': ['Smith', 'Jones'],
        'CreditScore': [650, 700],
        'Geography': ['France', 'Spain'],
        'Gender': ['Male', 'Female'],
        'Age': [35, 42],
        'Tenure': [3, 5],
        'Balance': [10000.0, 0.0],
        'NumOfProducts': [1, 2],
        'HasCrCard': [1, 1],
        'IsActiveMember': [1, 0],
        'EstimatedSalary': [50000.0, 75000.0],
        'Exited': [0, 1]
    })

# Optional: Configure pytest behavior
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )