"""Step 2 end to end: mask -> normalize lighting -> apply mask -> resize. process.md Step 2.

`preprocessing.masking_method` picks the mask:
  - "background_grabcut" (default): background-model + GrabCut, see background.py.
  - "otsu": the original method, kept for comparison. Fails on the real
    photos (see background.py docstring).

`preprocessing.lighting_normalization` scales each photo so its grey paper
matches one reference grey before features are extracted (normalization.py,
ADR-013 item 1). The mask is computed on the raw photo; normalization does
not move it.
"""

import numpy as np

from copra_grading.preprocessing.background import mask_image_background
from copra_grading.preprocessing.normalization import normalize_lighting
from copra_grading.preprocessing.otsu import apply_mask, mask_image, resize_masked


def compute_mask(image: np.ndarray, config: dict) -> np.ndarray:
    pre = config["preprocessing"]
    method = pre["masking_method"]
    if method == "background_grabcut":
        return mask_image_background(
            image,
            threshold_k=pre.get("background_threshold_k", 4.0),
            grabcut_iters=pre.get("grabcut_iters", 3),
        )
    if method == "otsu":
        return mask_image(image)
    raise ValueError(f"unknown masking_method: {method!r}")


def normalize(image: np.ndarray, config: dict) -> np.ndarray:
    """Grey-paper lighting normalization if enabled in config, else `image` unchanged."""
    norm = config["preprocessing"].get("lighting_normalization", {})
    if not norm.get("enabled", False):
        return image
    return normalize_lighting(image, norm["reference_grey"])


def masked_crop(image: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """RGB photo -> (normalized, cropped, background-zeroed image; full-frame mask)."""
    mask = compute_mask(image, config)
    return apply_mask(normalize(image, config), mask), mask


def preprocess(image: np.ndarray, config: dict) -> np.ndarray:
    """RGB photo -> cropped, background-zeroed, normalized, fixed-size masked image."""
    crop, _ = masked_crop(image, config)
    return resize_masked(crop, config["preprocessing"]["resize_to"])
