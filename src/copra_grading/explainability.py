"""TreeSHAP explainability. process.md Step 8 (and the per-photo view in Step 9).

Runs on the selected model. Importance is summed by feature group: texture
(GLCM), color (HSV/LAB), edge (Canny). Grouping is also what keeps the
result readable: TreeSHAP is unreliable per individual feature when
features are strongly correlated (thesis §2.4.1), e.g. one GLCM property at
twelve distance/angle combinations.

`model` is a models.FittedModel; SHAP runs on its underlying estimator.
"""

import numpy as np
import pandas as pd
import shap

FAMILIES = ("texture", "color", "edge")
_PREFIXES = {"glcm_": "texture", "hsv_": "color", "lab_": "color", "edge_": "edge"}


def feature_family(column: str) -> str:
    """Feature family of one feature column, from its name prefix."""
    for prefix, family in _PREFIXES.items():
        if column.startswith(prefix):
            return family
    raise ValueError(f"feature {column!r} belongs to no family ({sorted(_PREFIXES)})")


def feature_family_map(columns) -> dict[str, str]:
    return {c: feature_family(c) for c in columns}


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values as (n_rows, n_features, n_classes), whatever the library returns."""
    values = shap.TreeExplainer(model.estimator).shap_values(X)
    if isinstance(values, list):  # older per-class list form
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:  # binary models return one class's values only
        values = values[..., None]
    return values


def _by_family(per_feature: np.ndarray, columns, feature_family_map: dict[str, str]) -> dict[str, float]:
    out = {f: 0.0 for f in FAMILIES}
    for value, column in zip(per_feature, columns):
        out[feature_family_map[column]] += float(value)
    return out


# --- treeshap ---


def compute_aggregate_shap_by_family(model, X: pd.DataFrame, feature_family_map: dict[str, str]) -> dict[str, float]:
    """Return mean |SHAP value| summed per feature group.

    Mean is over rows and classes; each feature's mean |SHAP| is then summed
    into its family.
    """
    values = _shap_values(model, X)
    per_feature = np.abs(values).mean(axis=(0, 2))
    return _by_family(per_feature, X.columns, feature_family_map)


def explain_single_prediction(model, x: pd.Series, feature_family_map: dict[str, str]) -> dict[str, float]:
    """SHAP explanation for one uploaded photo's features, grouped by feature type.

    Signed contribution of each family toward the predicted class (positive
    pushes toward it).
    """
    row = x.to_frame().T.astype(float)
    predicted = int(np.asarray(model.estimator.predict(row)).astype(int)[0])
    values = _shap_values(model, row)[0, :, predicted]
    return _by_family(values, row.columns, feature_family_map)


def family_shares(contributions: dict[str, float]) -> dict[str, float]:
    """|contribution| of each family as a share of the total (sums to 1)."""
    total = sum(abs(v) for v in contributions.values())
    if total == 0:
        return {f: 0.0 for f in contributions}
    return {f: abs(v) / total for f, v in contributions.items()}
