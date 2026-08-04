"""TreeSHAP against the selected algorithm's combined all-angle version only. Spec §10.

Not run against the single-angle deployment version, and not run separately
for all three ensemble candidates. Aggregate resulting feature-importance by
family (texture/GLCM, color/HSV-LAB, edge/Canny), not per individual feature -
answers "how much did texture vs. color vs. edges matter overall."

Also exposes the per-classification live SHAP function the Streamlit
deployment layer calls for one submitted image at inference time - a distinct
computation from the aggregate research finding above, but implemented here
so deployment can import it directly.
"""

import pandas as pd


def compute_aggregate_shap_by_family(model, X: pd.DataFrame, feature_family_map: dict[str, str]) -> dict[str, float]:
    """Return mean |SHAP value| aggregated per feature family."""
    raise NotImplementedError


def explain_single_prediction(model, x: pd.Series) -> dict[str, float]:
    """Live, per-classification SHAP explanation for one submitted image's features."""
    raise NotImplementedError
