"""GLCM texture features. process.md Step 3a. Distance/angle set: see ADR-006 / configs/default.yaml.

Physical rationale: wet copra has a smoother, more uniform surface; drying
roughens and cracks it. Contrast/homogeneity/energy/entropy quantify that.

Never apply photometric augmentation anywhere in this pipeline
(see ../../../CLAUDE.md).
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from copra_grading.features import EmptyRegionError


def _region(
    gray: np.ndarray, foreground: np.ndarray, boundary_margin_px: int
) -> np.ndarray:
    """Boolean copra region, optionally eroded to drop resize-blended border pixels."""
    if foreground.shape != gray.shape:
        raise ValueError(
            f"foreground shape {foreground.shape} does not match image {gray.shape}"
        )
    region = foreground.astype(bool)
    if not region.any():
        raise EmptyRegionError("empty foreground mask: no copra pixels found")

    if boundary_margin_px > 0:
        k = 2 * boundary_margin_px + 1
        eroded = cv2.erode(region.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        # A tiny copra region can vanish under erosion; keep the uneroded mask then.
        if eroded.any():
            region = eroded
    return region


def extract_glcm_features(
    masked_image: np.ndarray,
    distances: list[int],
    angles_deg: list[int],
    foreground: np.ndarray | None = None,
    levels: int = 256,
    boundary_margin_px: int = 0,
) -> dict[str, float]:
    """Return contrast, homogeneity, energy, entropy per distance/angle combo.

    Each distance/angle pair is kept as a separate feature (not averaged) per
    ADR-006's documented default.

    `masked_image` is a 2-D uint8 grayscale array.

    Distance d means d pixel steps along the direction: (0, d) at 0 deg,
    (d, d) diagonally at 45/135 deg. scikit-image instead rounds
    d*sin/d*cos, so its d=1 and d=2 both land on a 1-pixel diagonal step
    and give identical features; diagonal distances are therefore passed
    as d*sqrt(2), which rounds to exactly d steps each way. Angles must be
    multiples of 45 degrees for this reason.

    `foreground` (bool, same shape) restricts the GLCM to copra-copra pixel
    pairs. Without it, the zeroed background and the cliff at the copra
    outline count as texture: the statistics are distorted (background-
    background pairs dilute contrast, outline pairs spike it) and depend on
    how much of the frame the copra fills rather than on the surface.
    With it, copra gray levels are
    shifted up so background can be level 0, and row/column 0 of the
    co-occurrence matrix (every pair touching background) is dropped before
    normalizing. Gray-level differences - all the stats depend on - are
    unchanged by the shift (at levels=256, copra levels 0 and 1 merge).

    `levels` (2-256, default 256 = unchanged behaviour) quantizes gray values
    into that many bins before counting. A copra region of a few thousand
    pixels spread over 256 levels gives a very sparse matrix; 32 or 64 levels
    usually gives steadier statistics. Treat it as a tunable (ADR-006).

    `boundary_margin_px` (default 0) erodes `foreground` before use. Leave it
    at 0 if the caller already eroded the mask (features.exclude_boundary),
    otherwise the border gets trimmed twice.

    Raises ValueError on an empty foreground or when a distance/angle pair
    has no valid pixel pairs, so cleaning can drop and report the row
    instead of letting a silent all-zero feature row through.
    """
    if masked_image.ndim != 2:
        raise ValueError(
            "extract_glcm_features expects a 2D grayscale array of shape "
            f"(H, W), got shape {masked_image.shape}"
        )
    if masked_image.dtype != np.uint8:
        raise ValueError(
            "extract_glcm_features expects a uint8 array with 0-255 gray "
            f"levels, got dtype {masked_image.dtype}"
        )
    if not distances or not angles_deg:
        raise ValueError("distances and angles_deg must both be non-empty")
    if any(d < 1 for d in distances):
        raise ValueError(f"distances must be >= 1 pixel, got {distances}")
    bad_angles = [a for a in angles_deg if a % 45 != 0]
    if bad_angles:
        raise ValueError(f"angles must be multiples of 45 degrees, got {bad_angles}")
    if not 2 <= levels <= 256:
        raise ValueError(f"levels must be between 2 and 256, got {levels}")

    gray = masked_image
    if levels < 256:
        gray = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)  # 0..levels-1

    if foreground is not None:
        region = _region(gray, foreground, boundary_margin_px)
        if levels == 256:
            gray = np.where(region, np.maximum(gray, 1), 0).astype(np.uint8)
            n_levels = 256
        else:
            gray = np.where(region, gray + 1, 0).astype(np.uint8)  # copra -> 1..levels
            n_levels = levels + 1
    else:
        n_levels = levels

    features: dict[str, float] = {}
    for d in distances:
        for a_deg in angles_deg:
            angle_rad = np.deg2rad(a_deg)
            step = d if a_deg % 90 == 0 else d * np.sqrt(2)
            counts = graycomatrix(
                gray,
                distances=[step],
                angles=[angle_rad],
                levels=n_levels,
                symmetric=True,
                normed=False,
            ).astype(np.float64)
            if foreground is not None:
                counts = counts[1:, 1:]
            total = counts.sum()
            if total <= 0:
                raise EmptyRegionError(
                    f"no valid pixel pairs at distance {d}, angle {a_deg}: "
                    "copra region too small or distance too large"
                )
            glcm = counts / total

            probs = glcm[:, :, 0, 0]
            nonzero_probs = probs[probs > 0]
            # "+ 0.0" turns -0.0 (single-entry matrix) into 0.0 for clean CSVs.
            entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs))) + 0.0

            # skimage's 'energy' prop is sqrt(ASM); the spec defines energy
            # as the sum of squared co-occurrence probabilities (ASM itself).
            asm = float(np.sum(probs**2))

            key_prefix = f"glcm_d{d}_a{a_deg}"
            features[f"{key_prefix}_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
            features[f"{key_prefix}_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
            features[f"{key_prefix}_energy"] = asm
            features[f"{key_prefix}_entropy"] = entropy

    return features
