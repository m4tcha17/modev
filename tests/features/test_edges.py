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


def test_foreground_ignores_outline_edges():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 10:30] = 180
    fg = image > 0

    with_outline = extract_edge_features(image, 50, 150)
    interior_only = extract_edge_features(image, 50, 150, foreground=fg, boundary_margin_px=3)

    assert with_outline["edge_density"] > 0
    assert interior_only["edge_density"] == 0.0


def test_foreground_keeps_interior_crack():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[5:35, 5:35] = 180
    image[5:35, 19:21] = 40  # dark crack down the middle
    fg = image > 0

    feats = extract_edge_features(image, 50, 150, foreground=fg, boundary_margin_px=3)

    assert feats["edge_density"] > 0
