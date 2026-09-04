import numpy as np
import pytest

from copra_grading.features.glcm import extract_glcm_features


def test_constant_image_has_near_zero_contrast_and_entropy():
    gray = np.full((16, 16), 128, dtype=np.uint8)

    features = extract_glcm_features(gray, distances=[1], angles_deg=[0])

    assert features["glcm_d1_a0_contrast"] == 0.0
    assert features["glcm_d1_a0_entropy"] == 0.0
    assert features["glcm_d1_a0_energy"] == 1.0


def test_checkerboard_image_has_higher_contrast_than_constant_image():
    checkerboard = np.zeros((16, 16), dtype=np.uint8)
    checkerboard[::2, ::2] = 255
    checkerboard[1::2, 1::2] = 255
    constant = np.full((16, 16), 128, dtype=np.uint8)

    checker_features = extract_glcm_features(checkerboard, distances=[1], angles_deg=[0])
    constant_features = extract_glcm_features(constant, distances=[1], angles_deg=[0])

    assert checker_features["glcm_d1_a0_contrast"] > constant_features["glcm_d1_a0_contrast"]


def test_returns_one_set_of_stats_per_distance_angle_combination():
    gray = np.full((16, 16), 128, dtype=np.uint8)

    features = extract_glcm_features(gray, distances=[1, 2], angles_deg=[0, 90])

    expected_keys = {
        f"glcm_d{d}_a{a}_{stat}"
        for d in (1, 2)
        for a in (0, 90)
        for stat in ("contrast", "homogeneity", "energy", "entropy")
    }
    assert set(features.keys()) == expected_keys


def test_raises_value_error_on_non_2d_input():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        extract_glcm_features(rgb, distances=[1], angles_deg=[0])


def test_raises_value_error_on_non_uint8_input():
    gray = np.full((16, 16), 128.7, dtype=np.float32)

    with pytest.raises(ValueError):
        extract_glcm_features(gray, distances=[1], angles_deg=[0])
