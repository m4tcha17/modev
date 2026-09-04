"""GLCM texture features. Spec §5a. Distance/angle set: see ADR-006 / configs/default.yaml.

Physical rationale: wet copra has a smoother, more uniform surface; drying
roughens and cracks it. Contrast/homogeneity/energy/entropy quantify that.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    masked_image: np.ndarray,
    distances: list[int],
    angles_deg: list[int],
) -> dict[str, float]:
    """Return contrast, homogeneity, energy, entropy per distance/angle combo.

    Each distance/angle pair is kept as a separate feature (not averaged) per
    ADR-006's documented default.
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
    gray = masked_image
    # CAVEAT: masked_image's background pixels are zeroed by apply_mask, and
    # that zeroed region falls inside this measurement. The intensity cliff
    # at the object/background boundary inflates GLCM contrast (observed
    # ~4x higher than an unmasked equivalent), so this partly measures the
    # object's silhouette rather than only its surface texture. Whether/how
    # to correct this (e.g. eroding the mask before measurement) is a
    # deferred research-methodology decision - see the final-review section
    # of
    # .superpowers/sdd/2026-08-04-preprocessing-features-pipeline/progress.md
    features: dict[str, float] = {}

    for d in distances:
        for a_deg in angles_deg:
            angle_rad = np.deg2rad(a_deg)
            glcm = graycomatrix(
                gray,
                distances=[d],
                angles=[angle_rad],
                levels=256,
                symmetric=True,
                normed=True,
            )
            probs = glcm[:, :, 0, 0]
            nonzero_probs = probs[probs > 0]
            entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

            # skimage's 'energy' prop is sqrt(ASM); the spec defines energy
            # as the sum of squared co-occurrence probabilities (ASM itself).
            asm = graycoprops(glcm, "ASM")[0, 0]

            key_prefix = f"glcm_d{d}_a{a_deg}"
            features[f"{key_prefix}_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
            features[f"{key_prefix}_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
            features[f"{key_prefix}_energy"] = float(asm)
            features[f"{key_prefix}_entropy"] = entropy

    return features
