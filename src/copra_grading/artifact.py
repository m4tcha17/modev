"""Deployable artifact serialization. process.md Step 9. Format: see ADR-005.

Bundles the selected model (selection.py) with its exact preprocessing/
feature-extraction config so the Streamlit app reproduces Steps 2-3 on one
uploaded photo identically.

Layout of an artifact directory:
  manifest.json   model name, model file, class letters, feature order,
                  tuned params, the preprocessing + features config
  model.json      XGBoost native save (XGBClassifier.save_model), or
  model.joblib    joblib dump of the estimator (scikit-learn models and
                  LightGBM - ADR-005 allows LightGBM's .pkl save)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from copra_grading.models import FittedModel

MANIFEST = "manifest.json"
# Config sections inference needs to rebuild the training-time features.
INFERENCE_SECTIONS = ("preprocessing", "features")


@dataclass(frozen=True)
class Artifact:
    model: FittedModel
    feature_columns: list[str]
    config: dict  # INFERENCE_SECTIONS only
    params: dict
    extra: dict  # anything else the writer recorded (CV summary, dates)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Class letters for rows of `features`, reordered to the training columns."""
        return self.model.predict(features[self.feature_columns])


# --- serialize ---


def save_artifact(
    model: FittedModel,
    feature_columns: list[str],
    config: dict,
    out_dir: Path,
    params: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Serialize model (joblib for sklearn-compatible, native save for XGBoost/LightGBM
    - see ADR-005) plus feature_config to out_dir. Return the artifact path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model.name == "xgboost":
        model_file = "model.json"
        model.estimator.save_model(out_dir / model_file)
    else:
        model_file = "model.joblib"
        joblib.dump(model.estimator, out_dir / model_file)
    manifest = {
        "model_name": model.name,
        "model_file": model_file,
        "classes": [str(c) for c in model.classes],
        "feature_columns": list(feature_columns),
        "params": params or {},
        "config": {k: config[k] for k in INFERENCE_SECTIONS},
        "extra": extra or {},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_artifact(path: Path) -> Artifact:
    """Load a previously serialized model + config bundle for inference.

    Raises FileNotFoundError if the directory, manifest or model file is
    missing - the app must refuse to classify rather than fall back.
    """
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text())
    model_path = path / manifest["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"model file missing: {model_path}")
    if manifest["model_name"] == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier()
        estimator.load_model(model_path)
    else:
        estimator = joblib.load(model_path)
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(manifest["classes"], dtype=object)
    return Artifact(
        model=FittedModel(manifest["model_name"], estimator, encoder),
        feature_columns=manifest["feature_columns"],
        config=manifest["config"],
        params=manifest["params"],
        extra=manifest["extra"],
    )
