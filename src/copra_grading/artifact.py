"""Final deployable artifact serialization. Spec M-BM-'12. Format: see ADR-005.

Bundles the angle-retrained deployment model (selection/deployment_config.py's
output) with its exact preprocessing/feature-extraction config - never the
combined all-angle model from selection/algorithm_selection.py.
"""

from pathlib import Path

# --- serialize ---


def save_artifact(model, feature_config: dict, deployment_angle_config: str, out_dir: Path) -> Path:
    """Serialize model (joblib for sklearn-compatible, native save for XGBoost/LightGBM
    - see ADR-005) plus feature_config to out_dir. Return the artifact path.
    """
    raise NotImplementedError


def load_artifact(path: Path):
    """Load a previously serialized model + config bundle for inference."""
    raise NotImplementedError
