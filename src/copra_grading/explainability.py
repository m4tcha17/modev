"""TreeSHAP explainability. Spec §10."""

import pandas as pd

# --- treeshap ---


def compute_aggregate_shap_by_family(model, X: pd.DataFrame, feature_family_map: dict[str, str]) -> dict[str, float]:
    """Return mean |SHAP value| aggregated per feature family."""
    raise NotImplementedError


def explain_single_prediction(model, x: pd.Series) -> dict[str, float]:
    """Live, per-classification SHAP explanation for one submitted image's features."""
    raise NotImplementedError
