"""Canny edge/contour density features. Spec §5c.

Physical rationale: as copra dries and shrinks, its surface develops more
visible cracks/fissures. Edge density is a proxy for that structural change.
"""

import numpy as np


def extract_edge_features(
    masked_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, float]:
    """Return edge-pixel proportion and/or contour distribution statistics."""
    raise NotImplementedError
