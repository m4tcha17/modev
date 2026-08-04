"""HSV/LAB color-space features. Spec §5b.

Physical rationale: moisture loss shifts copra from pale/white toward
tan/brown. HSV separates hue/saturation from brightness (less confounded by
uncontrolled ambient lighting than raw RGB); LAB is perceptually uniform and
matches the gray-to-brown drying shift.

Never apply photometric augmentation anywhere in this pipeline - it corrupts
this signal directly (see ../../../CLAUDE.md).
"""

import numpy as np


def extract_color_features(masked_image: np.ndarray) -> dict[str, float]:
    """Return per-channel statistics (mean, std, and/or histogram stats) for HSV and LAB."""
    raise NotImplementedError
