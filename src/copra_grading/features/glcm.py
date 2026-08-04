"""GLCM texture features. Spec §5a. Distance/angle set: see ADR-006 / configs/default.yaml.

Physical rationale: wet copra has a smoother, more uniform surface; drying
roughens and cracks it. Contrast/homogeneity/energy/entropy quantify that.
"""

import numpy as np


def extract_glcm_features(
    masked_image: np.ndarray,
    distances: list[int],
    angles_deg: list[int],
) -> dict[str, float]:
    """Return contrast, homogeneity, energy, entropy per distance/angle combo.

    Each distance/angle pair is kept as a separate feature (not averaged) per
    ADR-006's documented default.
    """
    raise NotImplementedError
