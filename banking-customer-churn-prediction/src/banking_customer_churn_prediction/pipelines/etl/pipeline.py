"""ETL pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import extract_raw, clean_and_prepare_data

def create_pipeline(**kwargs) -> Pipeline:
    """Create the ETL pipeline."""
    return Pipeline(
        [
            node(
                func=extract_raw,
                inputs="raw_churn",
                outputs=["raw_df", "raw_data_quality_metrics"],  # ✅ CHANGED: Added metrics output
                name="extract_raw_data",
                tags=["etl", "validation"],
            ),
            node(
                func=clean_and_prepare_data,
                inputs=[
                    "raw_df", 
                    "raw_data_quality_metrics",
                    "params:data.columns.drop",  # ✅ CHANGED: Parameterized
                    "params:data.random_state"
                ],
                outputs="cleaned_customers",
                name="clean_and_prepare_data",
                tags=["etl", "cleaning"],
            ),
        ]
    )