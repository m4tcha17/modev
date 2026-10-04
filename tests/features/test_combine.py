import numpy as np

from copra_grading.features.combine import extract_all_features


def _config():
    return {
        "features": {
            "glcm": {"distances": [1], "angles_deg": [0, 90]},
            "canny": {"low_threshold": 50, "high_threshold": 150},
        }
    }


def test_combines_all_three_feature_families():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_all_features(image, _config())

    assert any(key.startswith("glcm_") for key in features)
    assert any(key.startswith("hsv_") for key in features)
    assert any(key.startswith("lab_") for key in features)
    assert "edge_density" in features
    assert all(isinstance(value, float) for value in features.values())


def test_feature_ordering_is_deterministic_across_calls():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    first_call = list(extract_all_features(image, _config()).keys())
    second_call = list(extract_all_features(image, _config()).keys())

    assert first_call == second_call


def test_feature_order_is_glcm_then_color_then_edge():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    keys = list(extract_all_features(image, _config()).keys())

    def family(key: str) -> str:
        if key.startswith("glcm_"):
            return "glcm"
        if key.startswith("hsv_") or key.startswith("lab_"):
            return "color"
        if key.startswith("edge_"):
            return "edge"
        raise AssertionError(f"unexpected key prefix: {key!r}")

    families_in_order = [family(key) for key in keys]

    # Each family must appear as one contiguous block, and the blocks must
    # occur in this fixed sequence: GLCM, then color, then edge. This is the
    # ordering a later artifact-serialization stage relies on for
    # deployment-time reproducibility.
    assert families_in_order == sorted(
        families_in_order, key=["glcm", "color", "edge"].index
    )
    assert set(families_in_order) == {"glcm", "color", "edge"}
    assert families_in_order[0] == "glcm"
    assert families_in_order[-1] == "edge"
