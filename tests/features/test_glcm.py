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


def test_foreground_excludes_outline_from_texture():
    # Flat copra on zeroed background: no surface texture at all. Only the
    # outline cliff can create contrast.
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 150
    fg = image > 0

    with_outline = extract_glcm_features(image, distances=[1], angles_deg=[0])
    interior_only = extract_glcm_features(image, distances=[1], angles_deg=[0], foreground=fg)

    assert with_outline["glcm_d1_a0_contrast"] > 0
    assert interior_only["glcm_d1_a0_contrast"] == 0.0
    assert interior_only["glcm_d1_a0_energy"] == 1.0


def test_foreground_keeps_real_texture():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 100
    image[8:24, 8:24:2] = 200  # stripes inside the copra
    fg = image > 0

    feats = extract_glcm_features(image, distances=[1], angles_deg=[0], foreground=fg)

    assert feats["glcm_d1_a0_contrast"] == pytest.approx(100.0**2)


def test_diagonal_distances_give_distinct_features():
    # scikit-image's own rounding makes d=1 and d=2 identical on diagonals.
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (40, 40)).astype(np.uint8)

    feats = extract_glcm_features(gray, distances=[1, 2, 3], angles_deg=[45, 135])

    for a in (45, 135):
        values = [feats[f"glcm_d{d}_a{a}_contrast"] for d in (1, 2, 3)]
        assert len(set(values)) == 3


def test_diagonal_distance_is_d_steps_each_way():
    # Stripes repeating every 2 pixels along both axes: a (2, 2) step lands
    # on the same value (contrast 0), a (1, 1) step doesn't.
    yy, xx = np.mgrid[0:20, 0:20]
    gray = np.where((yy + xx) % 4 < 2, 50, 200).astype(np.uint8)

    feats = extract_glcm_features(gray, distances=[1, 2], angles_deg=[45])

    assert feats["glcm_d2_a45_contrast"] == 0.0
    assert feats["glcm_d1_a45_contrast"] > 0.0
