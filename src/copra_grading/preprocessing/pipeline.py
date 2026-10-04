"""Step 2 end to end: mask -> apply mask -> resize. process.md Step 2.

`preprocessing.masking_method` picks the mask:
  - "background_grabcut" (default): background-model + GrabCut, see background.py.
  - "otsu": the original method, kept for comparison. Fails on the real
    photos (see background.py docstring).
"""

import numpy as np

from copra_grading.preprocessing.background import mask_image_background
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


def preprocess(image: np.ndarray, config: dict) -> np.ndarray:
    """RGB photo -> cropped, background-zeroed, fixed-size masked image."""
    mask = compute_mask(image, config)
    return resize_masked(apply_mask(image, mask), config["preprocessing"]["resize_to"])
