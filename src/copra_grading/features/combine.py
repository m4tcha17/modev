"""Concatenate GLCM + color + edge features into one row per angle image. Spec §5."""

import numpy as np

from copra_grading.features.color import extract_color_features
from copra_grading.features.edges import extract_edge_features
from copra_grading.features.glcm import extract_glcm_features


def extract_all_features(masked_image: np.ndarray, config: dict) -> dict[str, float]:
    """Run all three feature families and concatenate into one dict.

    Feature ordering here must match whatever ordering artifact/serialize.py
    records - deployment-time extraction has to reproduce it exactly.
    """
    raise NotImplementedError
