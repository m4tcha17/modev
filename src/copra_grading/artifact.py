"""Deployable artifact serialization. process.md Step 9. Format: see ADR-005.

Bundles the selected model (selection.py) with its exact preprocessing/
feature-extraction config so the Streamlit app reproduces Steps 2-3 on one
uploaded photo identically.
"""

from pathlib import Path

# --- serialize ---


def save_artifact(model, feature_config: dict, out_dir: Path) -> Path:
    """Serialize model (joblib for sklearn-compatible, native save for XGBoost/LightGBM
    - see ADR-005) plus feature_config to out_dir. Return the artifact path.
    """
    raise NotImplementedError


def load_artifact(path: Path):
    """Load a previously serialized model + config bundle for inference."""
    raise NotImplementedError
