import numpy as np

from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import apply_mask, mask_image


def _config():
    return {
        "features": {
            "glcm": {"distances": [1, 2], "angles_deg": [0, 90]},
            "canny": {"low_threshold": 50, "high_threshold": 150},
        }
    }


def _synthetic_tray_image(size=60, tan_rgb=(200, 170, 120), tray_rgb=(60, 60, 70)):
    """Tan copra disk, centered, covering ~69% of the frame, on a darker tray."""
    image = np.full((size, size, 3), tray_rgb, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    radius = size * 0.47  # pi * 0.47^2 ~= 69% of the square frame's area
    disk_mask = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    image[disk_mask] = tan_rgb
    return image, disk_mask


def test_mask_apply_extract_pipeline_selects_object_and_produces_finite_features():
    image, disk_mask = _synthetic_tray_image()

    mask = mask_image(image)

    # The mask must pick the object (disk), not the tray background. The
    # disk covers > 50% of the frame and both are centered, so this also
    # exercises finding 1's centered-object-selection fix end-to-end.
    center = image.shape[0] // 2
    assert mask[center, center]
    assert not mask[0, 0]
    np.testing.assert_array_equal(mask, disk_mask)

    masked_image = apply_mask(image, mask)
    features = extract_all_features(masked_image, _config())

    for prefix in ("glcm_", "hsv_", "lab_", "edge_"):
        assert any(key.startswith(prefix) for key in features), f"missing {prefix} features"

    for key, value in features.items():
        assert isinstance(value, float), f"{key} is not a float"
        assert np.isfinite(value), f"{key} is not finite: {value}"
