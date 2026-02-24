import pandas as pd

def predict_single(
    input_df: pd.DataFrame,
    preprocessor,
    model
) -> float:
    """
    Run inference on a single record.
    Assumes preprocessor is already fitted.
    """
    X = preprocessor.transform(input_df)
    proba = model.predict_proba(X)[:, 1]
    return float(proba[0])
