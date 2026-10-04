"""Otsu background masking + resize. process.md Step 2.

NOT the default any more: on the real photos Otsu lumps the grey background
in with the white meat (see background.py, ADR-008). Kept for comparison and
because apply_mask/resize_masked here are shared by every method.

Every photo is taken with the same phone, at
a fixed 10 cm distance, same camera settings, on the same mid-grey
background, so one global threshold per image is sufficient - no per-photo
recalibration.

Fallback methods (adaptive threshold, HSV/saturation mask, GrabCut) are
documented in the spec as reach-for-if-needed, not preemptive - only implement
one if evaluation shows Otsu masks are unreliable under real field lighting.
Keep any fallback behind the same `mask_image` signature so feature extraction
never needs to know which method produced the mask.
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def foreground_mask(masked_image: np.ndarray) -> np.ndarray:
    """Copra pixels of a masked image: any channel nonzero (see apply_mask)."""
    if masked_image.ndim == 3:
        return np.any(masked_image != 0, axis=-1)
    return masked_image != 0


def mask_image(image: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating copra pixels from background.

    Steps (Step 2): grayscale conversion -> intensity histogram -> Otsu threshold
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
    # object. The camera setup is fixed and the sample is roughly centered in
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
    # Downstream code reads "any channel nonzero" as copra, so lift copra
    # pixels to at least 1 - a pure-black copra pixel would otherwise read
    # as background.
    cropped_image[cropped_mask] = np.maximum(cropped_image[cropped_mask], 1)
    cropped_image[~cropped_mask] = 0
    return cropped_image


def resize_masked(masked_image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a masked image to `size` = (width, height) so every photo is on
    the same scale.

    Nearest-neighbour interpolation: it never blends copra pixels with the
    zeroed background, so the nonzero-means-copra convention the feature
    extractors rely on still holds after resizing.
    """
    return cv2.resize(masked_image, tuple(size), interpolation=cv2.INTER_NEAREST)
