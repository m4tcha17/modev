"""Grey-paper lighting normalization. Thesis §4.3.2.2, REQ-4.1-6; ADR-013 item 1.

The capture-box lights dim slowly during a session, so the paper (and the
copra on it) is not equally bright in every photo. Each photo is scaled so
its paper matches one fixed reference grey before features are extracted:

    gain_c = reference_c / paper_c        (per RGB channel)

`paper_c` is the average of the background surface fitted to the border band
(background.paper_color), so the copra itself never influences the gain.
Per-channel gains correct a color cast as well as brightness - the paper is
neutral grey, so any tint it shows is the light's.

This is not photometric augmentation (see ../../../CLAUDE.md): it creates no
extra copies, and it applies the same deterministic correction to every
photo, at training and at inference. Values that would exceed 255 clip.
"""

import numpy as np

from copra_grading.preprocessing.background import paper_color


def lighting_gains(image: np.ndarray, reference_grey: float) -> np.ndarray:
    """Per-channel RGB gains that map this photo's paper onto the reference grey."""
    paper = np.maximum(paper_color(image), 1.0)
    return reference_grey / paper


def apply_gains(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Scale an RGB uint8 image per channel, rounding and clipping to 0-255."""
    scaled = image.astype(np.float32) * gains.astype(np.float32)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def normalize_lighting(image: np.ndarray, reference_grey: float) -> np.ndarray:
    """Return `image` (H, W, 3 RGB uint8) scaled so its paper is `reference_grey`."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"normalize_lighting expects an RGB image, got shape {image.shape}")
    return apply_gains(image, lighting_gains(image, reference_grey))
