"""Concatenate GLCM + color + edge features into one row per angle image. Spec §5."""

import cv2
import numpy as np

from copra_grading.features.color import extract_color_features
from copra_grading.features.edges import extract_edge_features
from copra_grading.features.glcm import extract_glcm_features


def extract_all_features(masked_image: np.ndarray, config: dict) -> dict[str, float]:
    """Run all three feature families and concatenate into one dict.

    Feature ordering here must match whatever ordering artifact/serialize.py
    records - deployment-time extraction has to reproduce it exactly.
    """
    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )

    glcm_config = config["features"]["glcm"]
    canny_config = config["features"]["canny"]

    features: dict[str, float] = {}
    features.update(
        extract_glcm_features(
            gray,
            distances=glcm_config["distances"],
            angles_deg=glcm_config["angles_deg"],
        )
    )
    features.update(extract_color_features(masked_image))
    features.update(
        extract_edge_features(
            gray,
            low_threshold=canny_config["low_threshold"],
            high_threshold=canny_config["high_threshold"],
        )
    )
    return features
