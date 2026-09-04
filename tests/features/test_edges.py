import numpy as np

from copra_grading.features.edges import extract_edge_features


def test_uniform_image_has_zero_edge_density():
    gray = np.full((20, 20), 128, dtype=np.uint8)

    features = extract_edge_features(gray, low_threshold=50, high_threshold=150)

    assert features["edge_density"] == 0.0


def test_half_black_half_white_image_has_positive_edge_density():
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255

    features = extract_edge_features(gray, low_threshold=50, high_threshold=150)

    assert features["edge_density"] > 0.0
    assert features["edge_contour_count"] >= 1
