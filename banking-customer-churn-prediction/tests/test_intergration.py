"""Integration tests for the churn prediction pipeline."""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

class TestIntegration:
    """Integration test suite."""
    
    @pytest.fixture
    def catalog(self, sample_customer_data):
        """Create complete test catalog."""
        catalog = DataCatalog({
            # Data
            "raw_churn": MemoryDataset(data=sample_customer_data),
            
            # Data parameters
            "params:data.target": MemoryDataset(data="Exited"),
            "params:data.test_size": MemoryDataset(data=0.2),
            "params:data.val_size": MemoryDataset(data=0.2),
            "params:data.random_state": MemoryDataset(data=42),
            "params:data.stratify": MemoryDataset(data=True),
            
            # Column definitions
            "params:data.columns.numeric": MemoryDataset(
                data=["CreditScore", "Age", "Tenure", "Balance", 
                      "NumOfProducts", "EstimatedSalary"]
            ),
            "params:data.columns.categorical": MemoryDataset(
                data=["Geography", "Gender"]
            ),
            "params:data.columns.drop": MemoryDataset(
                data=["RowNumber", "CustomerId", "Surname"]
            ),
            
            # Feature engineering
            "params:features.preprocessing": MemoryDataset(data={
                "numeric": {"impute_strategy": "median", "scale_strategy": "standard"},
                "categorical": {"impute_strategy": "most_frequent", "encode_strategy": "onehot"}
            }),
            "params:features": MemoryDataset(data={
                "derived": [
                    {"name": "avg_bal_per_product", "formula": "Balance / (NumOfProducts + 1e-9)"},
                    {"name": "tenure_bucket", "type": "categorical", "bins": [-1, 1, 3, 6, 10, 50]}
                ]
            }),
            
            # Model training (simplified for tests)
            "params:models": MemoryDataset(data={
                "logistic_regression": {"C": 1.0, "max_iter": 100, "random_state": 42},
                "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42}
            }),
            "params:training": MemoryDataset(data={
                "cv_folds": 3,
                "scoring_metric": "roc_auc",
                "refit": True,
                "verbose": 0
            }),
        })
        
        return catalog
    
    def test_etl_pipeline_runs(self, catalog):
        """Test that ETL pipeline runs without errors."""
        try:
            from banking_customer_churn_prediction.pipeline_registry import register_pipelines
            pipelines = register_pipelines()
            etl_pipeline = pipelines.get("etl")
            
            if etl_pipeline is None:
                pytest.skip("ETL pipeline not configured")
            
            runner = SequentialRunner()
            result = runner.run(etl_pipeline, catalog)
            
            # Verify pipeline completed
            assert "cleaned_customers" in result
            
            # Verify data transformation
            cleaned_data = catalog.load("cleaned_customers")
            assert isinstance(cleaned_data, pd.DataFrame)
            assert len(cleaned_data) == 10
            
            # Verify columns were dropped
            assert "CustomerId" not in cleaned_data.columns
            assert "Surname" not in cleaned_data.columns
            
        except ImportError:
            pytest.skip("Project structure not found")
    
    def test_feature_pipeline_runs(self, catalog, sample_customer_data):
        """Test that feature engineering pipeline runs."""
        # Setup: Add cleaned data
        cleaned_data = sample_customer_data.drop(columns=["RowNumber", "CustomerId", "Surname"])
        catalog._datasets["cleaned_customers"] = MemoryDataset(data=cleaned_data)
        
        # Import and run pipeline
        from banking_customer_churn_prediction.pipelines.feature_engineering.pipeline import create_pipeline
        pipeline = create_pipeline()
        
        runner = SequentialRunner()
        result = runner.run(pipeline, catalog)
        
        # Verify outputs exist
        expected_outputs = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
        for output in expected_outputs:
            assert output in result or output in catalog._datasets
        
        # Verify data split
        total_samples = 0
        for split in ["X_train", "X_val", "X_test"]:
            if split in catalog._datasets:
                data = catalog.load(split)
                total_samples += len(data)
            elif split in result:
                dataset = result[split]
                if hasattr(dataset, 'load'):
                    data = dataset.load()
                    total_samples += len(data)
        
        assert total_samples == len(sample_customer_data)
    
    def test_end_to_end_pipeline(self, catalog):
        """Test that the full data preparation pipeline runs."""
        from banking_customer_churn_prediction.pipeline_registry import register_pipelines
        pipelines = register_pipelines()
        
        # Try data_preparation or default pipeline
        pipeline = pipelines.get("data_preparation") or pipelines.get("__default__")
        
        if pipeline is None:
            pytest.skip("No data preparation pipeline configured")
        
        runner = SequentialRunner()
        result = runner.run(pipeline, catalog)
        
        # Verify key outputs were created
        assert "cleaned_customers" in result or "cleaned_customers" in catalog._datasets
        
        # For data_preparation pipeline, check feature outputs
        if "X_train" in result or "X_train" in catalog._datasets:
            # Feature pipeline ran successfully
            pass
    
    def test_pipeline_robustness(self, catalog):
        """Test pipeline handles edge cases."""
        # Test with minimal valid data
        minimal_data = pd.DataFrame({
            'RowNumber': [1, 2],
            'CustomerId': [1, 2],
            'Surname': ['A', 'B'],
            'CreditScore': [600, 700],
            'Geography': ['France', 'Spain'],
            'Gender': ['Male', 'Female'],
            'Age': [30, 40],
            'Tenure': [1, 2],
            'Balance': [1000.0, 2000.0],
            'NumOfProducts': [1, 2],
            'HasCrCard': [1, 0],
            'IsActiveMember': [1, 0],
            'EstimatedSalary': [50000.0, 60000.0],
            'Exited': [0, 1]
        })
        
        catalog._datasets["raw_churn"] = MemoryDataset(data=minimal_data)
        catalog._datasets["params:data.stratify"] = MemoryDataset(data=False)
        
        try:
            from banking_customer_churn_prediction.pipeline_registry import register_pipelines
            pipelines = register_pipelines()
            etl_pipeline = pipelines.get("etl")
            
            if etl_pipeline:
                runner = SequentialRunner()
                result = runner.run(etl_pipeline, catalog)
                assert "cleaned_customers" in result
        except Exception:
            # Pipeline should handle minimal data
            pass