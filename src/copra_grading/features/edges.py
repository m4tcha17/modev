"""Canny edge/contour density features. process.md Step 3c.

Physical rationale: as copra dries and shrinks, its surface develops more
visible cracks/fissures. Edge density is a proxy for that structural change.
"""

import cv2
import numpy as np


def extract_edge_features(
    masked_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
    foreground: np.ndarray | None = None,
    boundary_margin_px: int = 3,
) -> dict[str, float]:
    """Return edge density (edge pixels / copra pixels) and contour count.

    `foreground` (bool, same shape) limits measurement to the copra
    interior: the mask is eroded by `boundary_margin_px` and only edges
    inside it count, so the copra outline itself is not read as cracks.
    Without it, every nonzero pixel counts as copra and the outline is
    included.
    """
    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    if foreground is not None:
        region = foreground.astype(np.uint8)
        if boundary_margin_px > 0:
            k = 2 * boundary_margin_px + 1
            region = cv2.erode(region, np.ones((k, k), np.uint8))
        region = region.astype(bool)
        edges = np.where(region, edges, 0).astype(np.uint8)
        region_pixel_count = int(region.sum())
    else:
        region_pixel_count = int(np.count_nonzero(gray))

    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = edge_pixel_count / max(region_pixel_count, 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "edge_density": float(edge_density),
        "edge_contour_count": float(len(contours)),
    }
