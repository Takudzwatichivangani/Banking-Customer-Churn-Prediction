"""Postprocessing pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import (
    write_high_risk_to_csv,
    format_business_output,
    generate_summary_report,
    save_to_multiple_formats
)

def create_pipeline(**kwargs) -> Pipeline:
    """Create the postprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=write_high_risk_to_csv,
                inputs=["high_risk_df"],
                outputs="formatted_high_risk_df",
                name="format_high_risk_data",
                tags=["postprocessing", "formatting"],
            ),
            node(
                func=format_business_output,
                inputs=["formatted_high_risk_df"],
                outputs="business_ready_output",
                name="create_business_output",
                tags=["postprocessing", "business"],
            ),
            node(
                func=generate_summary_report,
                inputs=["business_ready_output"],
                outputs="summary_report",
                name="generate_summary_report",
                tags=["postprocessing", "reporting"],
            ),
            node(
                func=save_to_multiple_formats,
                inputs=["business_ready_output"],
                outputs="exported_files",
                name="export_multiple_formats",
                tags=["postprocessing", "export"],
            ),
        ]
    )