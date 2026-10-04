"""GLCM texture features. process.md Step 3a. Distance/angle set: see ADR-006 / configs/default.yaml.

Physical rationale: wet copra has a smoother, more uniform surface; drying
roughens and cracks it. Contrast/homogeneity/energy/entropy quantify that.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    masked_image: np.ndarray,
    distances: list[int],
    angles_deg: list[int],
    foreground: np.ndarray | None = None,
) -> dict[str, float]:
    """Return contrast, homogeneity, energy, entropy per distance/angle combo.

    Each distance/angle pair is kept as a separate feature (not averaged) per
    ADR-006's documented default.

    `foreground` (bool, same shape) restricts the GLCM to copra-copra pixel
    pairs. Without it, the zeroed background and the cliff at the copra
    outline count as texture: contrast is inflated and partly measures the
    silhouette instead of the surface. With it, copra gray levels are
    shifted to 1..255, background is set to level 0, and row/column 0 of the
    co-occurrence matrix (every pair touching background) is dropped before
    normalizing. Gray-level differences - all the stats depend on - are
    unchanged by the shift (except 0 and 1 merging).
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
    if foreground is not None:
        gray = np.where(foreground, np.maximum(masked_image, 1), 0).astype(np.uint8)
    else:
        gray = masked_image

    features: dict[str, float] = {}
    for d in distances:
        for a_deg in angles_deg:
            angle_rad = np.deg2rad(a_deg)
            counts = graycomatrix(
                gray,
                distances=[d],
                angles=[angle_rad],
                levels=256,
                symmetric=True,
                normed=False,
            ).astype(np.float64)
            if foreground is not None:
                counts = counts[1:, 1:]
            total = counts.sum()
            glcm = counts / total if total > 0 else counts

            probs = glcm[:, :, 0, 0]
            nonzero_probs = probs[probs > 0]
            entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

            # skimage's 'energy' prop is sqrt(ASM); the spec defines energy
            # as the sum of squared co-occurrence probabilities (ASM itself).
            asm = float(np.sum(probs**2))

            key_prefix = f"glcm_d{d}_a{a_deg}"
            features[f"{key_prefix}_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
            features[f"{key_prefix}_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
            features[f"{key_prefix}_energy"] = asm
            features[f"{key_prefix}_entropy"] = entropy

    return features
