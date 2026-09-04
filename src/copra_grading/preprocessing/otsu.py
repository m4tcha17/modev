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

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def mask_image(image: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating copra pixels from background.

    Steps (§4): grayscale conversion -> intensity histogram -> Otsu threshold
    -> binary mask. Downstream feature extraction must consume the masked
    image only, never the raw photo.

    `image` must be an (H, W) grayscale array or an (H, W, 3) RGB uint8
    array - not BGR. cv2.imread returns BGR by default and must be converted
    (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) before calling this function.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    threshold = threshold_otsu(gray)
    above = gray > threshold
    below = ~above
    # Otsu splits pixels into two classes but doesn't say which one is the
    # object. The camera is fixed and the sample is roughly centered in
    # frame (module docstring above), so whichever side contains the center
    # pixel is the foreground. This holds regardless of what fraction of the
    # frame the object covers - unlike a smaller-side-wins heuristic, which
    # silently inverts (returns the tray instead of the object) once the
    # object covers more than ~50% of the frame.
    center = (gray.shape[0] // 2, gray.shape[1] // 2)
    return above if above[center] else below


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop `image` to `mask`'s bounding box and zero out background pixels.

    This cropped, background-zeroed array is the "masked_image" every
    feature-extraction function downstream expects as input.
    """
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros_like(image)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_image = image[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1]
    cropped_image[~cropped_mask] = 0
    return cropped_image
