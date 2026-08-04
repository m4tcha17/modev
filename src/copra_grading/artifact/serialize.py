"""Serialize the deployment model + preprocessing/feature-extraction config. Spec §12.

Config bundled alongside the model must include: Otsu masking parameters, the
GLCM angle/distance set (ADR-006), exact feature ordering, and the chosen
deployment angle config (from selection/deployment_config.py) - everything a
separate Streamlit app needs to reproduce identical inference-time preprocessing.
"""

from pathlib import Path


def save_artifact(model, feature_config: dict, deployment_angle_config: str, out_dir: Path) -> Path:
    """Serialize model (joblib for sklearn-compatible, native save for XGBoost/LightGBM
    - see ADR-005) plus feature_config to out_dir. Return the artifact path.
    """
    raise NotImplementedError


def load_artifact(path: Path):
    """Load a previously serialized model + config bundle for inference."""
    raise NotImplementedError
