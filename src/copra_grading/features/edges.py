"""Canny edge/contour density features. process.md Step 3c.

Physical rationale: as copra dries and shrinks, its surface develops more
visible cracks/fissures. Edge density is a proxy for that structural change.

Notes
-----
- Thresholds are fixed absolute values from config on purpose: the capture
  setup is fixed (same phone, 10 cm, mid-grey background), and per-image
  adaptive thresholds would make edge density depend on brightness, which
  already carries the color signal.
- Never apply photometric augmentation anywhere in this pipeline
  (see ../../../CLAUDE.md).
"""

import cv2
import numpy as np


def _region(
    masked_image: np.ndarray,
    foreground: np.ndarray | None,
    boundary_margin_px: int,
) -> np.ndarray:
    """Boolean copra interior: foreground eroded so the outline is not read as cracks."""
    if foreground is not None:
        if foreground.shape != masked_image.shape[:2]:
            raise ValueError(
                f"foreground shape {foreground.shape} does not match "
                f"image {masked_image.shape[:2]}"
            )
        region = foreground.astype(bool)
    elif masked_image.ndim == 3:
        region = np.any(masked_image != 0, axis=-1)
    else:
        region = masked_image != 0

    if not region.any():
        raise ValueError("empty foreground mask: no copra pixels found")

    if boundary_margin_px > 0:
        k = 2 * boundary_margin_px + 1
        eroded = cv2.erode(region.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        # A tiny copra region can vanish under erosion; keep the uneroded mask then.
        if eroded.any():
            region = eroded

    return region


def extract_edge_features(
    masked_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
    foreground: np.ndarray | None = None,
    boundary_margin_px: int = 3,
    blur_ksize: int = 0,
) -> dict[str, float]:
    """Return edge density and contour statistics measured inside the copra.

    Keys:
      edge_density             edge pixels / copra pixels
      edge_contour_count       number of contours in the edge map
      edge_contour_density     contours per 1000 copra pixels (size-independent)
      edge_contour_mean_length mean contour arc length in pixels (0 if none)
      edge_contour_max_length  longest contour arc length in pixels (0 if none)

    `masked_image` is an (H, W, 3) RGB uint8 array (or an (H, W) uint8 gray
    image). `foreground` (bool, same HxW) limits measurement to the copra
    interior: the mask is eroded by `boundary_margin_px` and only edges inside
    it count, so the copra outline itself is not read as cracks. Without it the
    foreground is inferred from non-zero pixels and treated the same way.

    `blur_ksize` (odd int, 0 = off) applies a Gaussian blur before Canny to
    suppress sensor noise. Off by default so existing features are unchanged.

    Raises ValueError on an empty foreground so cleaning can drop and report
    the row instead of letting a silent all-zero feature row through.
    """
    if masked_image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {masked_image.dtype}")
    if masked_image.ndim == 3 and masked_image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got {masked_image.shape}")
    if masked_image.ndim not in (2, 3):
        raise ValueError(f"expected 2-D or 3-D image, got {masked_image.shape}")
    if blur_ksize and (blur_ksize < 3 or blur_ksize % 2 == 0):
        raise ValueError(f"blur_ksize must be 0 or an odd integer >= 3, got {blur_ksize}")

    region = _region(masked_image, foreground, boundary_margin_px)

    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )
    if blur_ksize:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = np.where(region, edges, 0).astype(np.uint8)

    region_pixel_count = int(region.sum())
    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = edge_pixel_count / region_pixel_count

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lengths = [cv2.arcLength(c, False) for c in contours]

    return {
        "edge_density": float(edge_density),
        "edge_contour_count": float(len(contours)),
        "edge_contour_density": float(len(contours) * 1000.0 / region_pixel_count),
        "edge_contour_mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "edge_contour_max_length": float(max(lengths)) if lengths else 0.0,
    }