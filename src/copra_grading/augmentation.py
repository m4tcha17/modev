"""Class-weighted geometric augmentation. Spec §7b. Multiplier: see ADR-002.

Geometric only (rotation, flip). No photometric augmentation, no SMOTE - see
../../CLAUDE.md.
"""

import numpy as np

# --- geometric ---


def augment_sample(
    image: np.ndarray, class_label: str, class_multipliers: dict[str, int]
) -> list[np.ndarray]:
    """Return `class_multipliers[class_label]` rotated/flipped copies of image."""
    raise NotImplementedError
