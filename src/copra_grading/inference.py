"""One-photo inference for the Streamlit app. process.md Step 9; SRS sections 4.4, 5.2, 5.3.

Runs the same Steps 2-3 as training, with the artifact's saved config:
background-model + GrabCut mask, grey-paper normalization, resize, then
GLCM + HSV/LAB + Canny features. Every failure raises UploadError with a
message for the user - the app never shows a grade that did not come from
a completed classification.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from copra_grading.artifact import Artifact
from copra_grading.dataset import CLASS_DESCRIPTIONS
from copra_grading.explainability import explain_single_prediction, family_shares, feature_family_map
from copra_grading.features import EmptyRegionError
from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import resize_masked
from copra_grading.preprocessing.pipeline import masked_crop

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# File signatures (magic bytes): the file type is judged by content, not name.
_SIGNATURES = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png"}


class UploadError(ValueError):
    """A photo that cannot be classified; the message is shown to the user."""


@dataclass(frozen=True)
class Classification:
    grade: str
    description: str
    features: pd.Series
    mask: np.ndarray  # full-frame boolean copra mask


def decode_upload(data: bytes) -> np.ndarray:
    """Validate an uploaded file and decode it to an (H, W, 3) RGB uint8 array."""
    if len(data) == 0:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"The file is {len(data) / 2**20:.1f} MB; the limit is {MAX_UPLOAD_BYTES // 2**20} MB.")
    if not any(data.startswith(sig) for sig in _SIGNATURES):
        raise UploadError("The file is not a JPEG or PNG image.")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise UploadError("The image is corrupted and could not be read.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def classify(artifact: Artifact, image: np.ndarray) -> Classification:
    """Grade one RGB photo of a cube's cut face with the deployed model."""
    config = artifact.config
    crop, mask = masked_crop(image, config)
    if not mask.any():
        raise UploadError("No copra was found in the photo. Use one cube on the grey paper, filling the frame centre.")
    masked = resize_masked(crop, config["preprocessing"]["resize_to"])
    try:
        features = pd.Series(extract_all_features(masked, config))
    except EmptyRegionError as err:
        raise UploadError(f"The copra region is too small to measure ({err}).") from err
    grade = str(artifact.predict(features.to_frame().T)[0])
    return Classification(grade, CLASS_DESCRIPTIONS.get(grade, ""), features, mask)


def explain(artifact: Artifact, features: pd.Series) -> dict[str, float]:
    """Share of the prediction's TreeSHAP contribution per feature family (sums to 1)."""
    x = features[artifact.feature_columns]
    contributions = explain_single_prediction(artifact.model, x, feature_family_map(artifact.feature_columns))
    return family_shares(contributions)
