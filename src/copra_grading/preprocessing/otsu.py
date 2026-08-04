"""Otsu background masking. Spec §4.

Primary/default masking method. Camera position is fixed across all six angle
shots of a sample (only the sample rotates), so one global threshold per image
is sufficient - no per-angle recalibration.

Fallback methods (adaptive threshold, HSV/saturation mask, GrabCut) are
documented in the spec as reach-for-if-needed, not preemptive - only implement
one if evaluation shows Otsu masks are unreliable under real field lighting.
Keep any fallback behind the same `mask_image` signature so feature extraction
never needs to know which method produced the mask.
"""

import numpy as np


def mask_image(image: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating copra pixels from background.

    Steps (§4): grayscale conversion -> intensity histogram -> Otsu threshold
    -> binary mask. Downstream feature extraction must consume the masked
    image only, never the raw photo.
    """
    raise NotImplementedError
