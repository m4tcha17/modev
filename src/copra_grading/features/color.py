"""HSV/LAB color-space features. process.md Step 3b.

Classes are moisture bands (docs/classes.md), and moisture shows up in
color: copra shifts from pale/white toward tan/brown as it dries.
HSV separates hue/saturation from brightness; LAB is perceptually uniform.
Only copra pixels count - the zeroed background is ignored.

Never apply photometric augmentation anywhere in this pipeline - it corrupts
this signal directly (see ../../../CLAUDE.md).

Notes
-----
- OpenCV uint8 LAB is scaled (L in 0-255, a/b offset by 128). Fine for tree
  models, but these are not standard CIELAB units.
- OpenCV hue is 0-179 and circular, so hue uses circular mean/std.
- Robust stats (median, p10, p90) are included next to mean/std so glare and
  shadow pixels don't dominate. The ablation can compare them.
"""

import cv2
import numpy as np

_HUE_TO_RAD = np.pi / 90.0  # OpenCV hue 0-179 -> 0-2*pi
_RAD_TO_HUE = 90.0 / np.pi


def _circular_hue_stats(hue_values: np.ndarray) -> tuple[float, float]:
    """Circular mean and std of OpenCV hue values, returned in OpenCV units (0-179)."""
    angles = hue_values * _HUE_TO_RAD
    sin_mean = float(np.sin(angles).mean())
    cos_mean = float(np.cos(angles).mean())
    resultant = min(float(np.hypot(sin_mean, cos_mean)), 1.0)

    mean_angle = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
    std_angle = np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))
    return float(mean_angle * _RAD_TO_HUE), float(std_angle * _RAD_TO_HUE)


def _foreground(
    masked_image: np.ndarray, mask: np.ndarray | None, erode_px: int
) -> np.ndarray:
    """Boolean copra-pixel mask, with the border eroded to drop resize-blended pixels."""
    if mask is not None:
        if mask.shape != masked_image.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} does not match image {masked_image.shape[:2]}"
            )
        foreground = mask.astype(bool)
    else:
        foreground = np.any(masked_image != 0, axis=-1)

    if not foreground.any():
        raise ValueError("empty foreground mask: no copra pixels found")

    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        eroded = cv2.erode(foreground.astype(np.uint8), kernel).astype(bool)
        # A tiny copra region can vanish under erosion; keep the uneroded mask then.
        if eroded.any():
            foreground = eroded

    return foreground


def extract_color_features(
    masked_image: np.ndarray,
    mask: np.ndarray | None = None,
    erode_px: int = 2,
) -> dict[str, float]:
    """Return per-channel statistics for HSV and LAB over copra pixels only.

    Per channel: mean, std, median, p10, p90. Hue uses circular mean/std
    (OpenCV units) plus median/p10/p90 of the raw values.

    `masked_image` must be an (H, W, 3) RGB uint8 array - not BGR. cv2.imread
    returns BGR by default and must be converted
    (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) before calling this function.

    `mask` is the real copra mask from preprocessing (H, W), preferred over
    inferring foreground from zero pixels. `erode_px` shrinks the foreground
    to skip border pixels blended with the zeroed background by resizing.

    Raises ValueError on an empty foreground so cleaning can drop and report
    the row instead of letting a silent all-zero feature row through.
    """
    if masked_image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {masked_image.dtype}")
    if masked_image.ndim != 3 or masked_image.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got {masked_image.shape}")

    foreground = _foreground(masked_image, mask, erode_px)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(masked_image, cv2.COLOR_RGB2LAB)

    features: dict[str, float] = {}
    for space_name, space_image in (("hsv", hsv), ("lab", lab)):
        for channel_idx, channel_name in enumerate(space_name):
            values = space_image[..., channel_idx][foreground].astype(np.float64)
            prefix = f"{space_name}_{channel_name}"

            if space_name == "hsv" and channel_name == "h":
                mean, std = _circular_hue_stats(values)
            else:
                mean, std = float(values.mean()), float(values.std())

            p10, median, p90 = np.percentile(values, [10, 50, 90])
            features[f"{prefix}_mean"] = mean
            features[f"{prefix}_std"] = std
            features[f"{prefix}_median"] = float(median)
            features[f"{prefix}_p10"] = float(p10)
            features[f"{prefix}_p90"] = float(p90)

    return features