import cv2
import numpy as np
import pytest

from copra_grading.features import EmptyRegionError
from copra_grading.features.color import extract_color_features


def test_uniform_foreground_color_has_zero_std_and_matches_cv2_conversion():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_color_features(image)

    expected_hsv = cv2.cvtColor(np.uint8([[[200, 100, 50]]]), cv2.COLOR_RGB2HSV)[0, 0]
    expected_lab = cv2.cvtColor(np.uint8([[[200, 100, 50]]]), cv2.COLOR_RGB2LAB)[0, 0]

    assert features["hsv_h_std"] == 0.0
    assert features["hsv_s_std"] == 0.0
    assert features["hsv_v_std"] == 0.0
    # Circular mean goes hue -> angle -> hue, so allow last-digit rounding.
    assert features["hsv_h_mean"] == pytest.approx(float(expected_hsv[0]))
    assert features["lab_l_mean"] == float(expected_lab[0])


def test_returns_mean_and_std_for_all_six_channels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_color_features(image)

    # No median/p10/p90 (ADR-009).
    expected_keys = {
        f"{space}_{channel}_{stat}"
        for space in ("hsv", "lab")
        for channel in space
        for stat in ("mean", "std")
    }
    assert set(features.keys()) == expected_keys


def test_hue_mean_wraps_around_the_color_wheel():
    # Half the pixels at hue 178, half at hue 2: neighbours on the wheel, so
    # the mean is ~0 (red), not 90 (cyan) as a linear mean would give.
    hsv = np.zeros((20, 20, 3), dtype=np.uint8)
    hsv[..., 1:] = 200
    hsv[:, :10, 0] = 178
    hsv[:, 10:, 0] = 2
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    features = extract_color_features(image, erode_px=0)

    hue_mean = features["hsv_h_mean"]
    assert min(hue_mean, 180 - hue_mean) < 1.0
    assert features["hsv_h_std"] < 5.0


def test_empty_foreground_raises_empty_region_error():
    with pytest.raises(EmptyRegionError):
        extract_color_features(np.zeros((20, 20, 3), dtype=np.uint8))
