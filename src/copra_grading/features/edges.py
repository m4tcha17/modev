"""Canny edge/contour density features. Spec §5c.

Physical rationale: as copra dries and shrinks, its surface develops more
visible cracks/fissures. Edge density is a proxy for that structural change.
"""

import cv2
import numpy as np


def extract_edge_features(
    masked_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, float]:
    """Return edge-pixel proportion and/or contour distribution statistics."""
    # CAVEAT: masked_image's background pixels are zeroed by apply_mask, and
    # that zeroed region falls inside this measurement. Canny picks up the
    # object/background boundary as a silhouette edge, so edge_density
    # largely measures the mask's outline shape rather than only surface
    # texture (cracks/fissures, per the module docstring above). Whether/how
    # to correct this (e.g. eroding the mask before measurement) is a
    # deferred research-methodology decision - see the final-review section
    # of
    # .superpowers/sdd/2026-08-04-preprocessing-features-pipeline/progress.md
    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    foreground_pixel_count = int(np.count_nonzero(gray))
    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = edge_pixel_count / max(foreground_pixel_count, 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "edge_density": float(edge_density),
        "edge_contour_count": float(len(contours)),
    }
