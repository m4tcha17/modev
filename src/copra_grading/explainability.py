"""TreeSHAP explainability. process.md Step 8 (and the per-photo view in Step 9).

Runs on the selected model. Importance is summed by feature group: texture
(GLCM), color (HSV/LAB), edge (Canny).
"""

import pandas as pd

# --- treeshap ---


def compute_aggregate_shap_by_family(model, X: pd.DataFrame, feature_family_map: dict[str, str]) -> dict[str, float]:
    """Return mean |SHAP value| summed per feature group."""
    raise NotImplementedError


def explain_single_prediction(model, x: pd.Series, feature_family_map: dict[str, str]) -> dict[str, float]:
    """SHAP explanation for one uploaded photo's features, grouped by feature type."""
    raise NotImplementedError
