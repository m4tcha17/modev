"""HSV/LAB color-space features. Spec §5b.

Physical rationale: moisture loss shifts copra from pale/white toward
tan/brown. HSV separates hue/saturation from brightness (less confounded by
uncontrolled ambient lighting than raw RGB); LAB is perceptually uniform and
matches the gray-to-brown drying shift.

Never apply photometric augmentation anywhere in this pipeline - it corrupts
this signal directly (see ../../../CLAUDE.md).
"""

import cv2
import numpy as np


def extract_color_features(masked_image: np.ndarray) -> dict[str, float]:
    """Return per-channel statistics (mean, std, and/or histogram stats) for HSV and LAB.

    `masked_image` must be an (H, W, 3) RGB uint8 array - not BGR. cv2.imread
    returns BGR by default and must be converted
    (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) before calling this function.
    """
    foreground_mask = np.any(masked_image != 0, axis=-1)
    if not foreground_mask.any():
        foreground_mask = np.ones(masked_image.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(masked_image, cv2.COLOR_RGB2LAB)

    features: dict[str, float] = {}
    for space_name, space_image in (("hsv", hsv), ("lab", lab)):
        for channel_idx, channel_name in enumerate(space_name):
            channel_values = space_image[..., channel_idx][foreground_mask].astype(np.float64)
            features[f"{space_name}_{channel_name}_mean"] = float(channel_values.mean())
            features[f"{space_name}_{channel_name}_std"] = float(channel_values.std())

    return features
