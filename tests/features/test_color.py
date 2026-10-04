import cv2
import numpy as np

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
    assert features["hsv_h_mean"] == float(expected_hsv[0])
    assert features["lab_l_mean"] == float(expected_lab[0])


def test_returns_mean_and_std_for_all_six_channels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_color_features(image)

    expected_keys = {
        f"{space}_{channel}_{stat}"
        for space in ("hsv", "lab")
        for channel in space
        for stat in ("mean", "std")
    }
    assert set(features.keys()) == expected_keys
