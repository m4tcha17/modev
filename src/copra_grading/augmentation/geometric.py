"""Rotation/flip augmentation with per-class multipliers. Spec §7b.

Must run only on training-fold data, after splitting (see splitting/groupkfold.py)
- never before, and never applied to validation/test folds.
"""

import numpy as np


def augment_sample(
    image: np.ndarray, class_label: str, class_multipliers: dict[str, int]
) -> list[np.ndarray]:
    """Return `class_multipliers[class_label]` rotated/flipped copies of image."""
    raise NotImplementedError
