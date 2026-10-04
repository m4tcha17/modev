"""Class-weighted geometric augmentation. process.md Step 4c. Multiplier: see ADR-002.

Training folds only, after the split. Geometric only (rotation, flip). No
photometric augmentation, no SMOTE - see ../../CLAUDE.md. Smaller classes get
more augmented copies than larger ones. Augmented images go back through
masking + feature extraction (Steps 2-3).
"""

import numpy as np

# --- geometric ---


def class_multipliers(class_counts: dict[str, int], max_multiplier: int) -> dict[str, int]:
    """Copies per original image for each class: the largest class gets the
    fewest, smaller classes proportionally more, capped at max_multiplier.
    """
    raise NotImplementedError


def augment_sample(
    image: np.ndarray, class_label: str, class_multipliers: dict[str, int]
) -> list[np.ndarray]:
    """Return `class_multipliers[class_label]` rotated/flipped copies of image."""
    raise NotImplementedError
