import numpy as np
import pytest

from copra_grading.preprocessing.background import paper_color
from copra_grading.preprocessing.normalization import apply_gains, lighting_gains, normalize_lighting
from copra_grading.preprocessing.pipeline import normalize, preprocess


def _copra_on_paper(paper_rgb, h=240, w=320, gradient=0.0):
    """Grey paper (optionally brighter at the bottom) with a white-meat block."""
    ramp = np.linspace(-gradient, gradient, h, dtype=np.float32)[:, None, None]
    image = np.broadcast_to(np.asarray(paper_rgb, np.float32), (h, w, 3)) + ramp
    image = image.copy()
    image[80:160, 100:220] = (200, 195, 180)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_paper_color_ignores_the_copra():
    image = _copra_on_paper((120, 128, 126))
    np.testing.assert_allclose(paper_color(image), (120, 128, 126), atol=1.0)


def test_paper_color_averages_a_lighting_gradient():
    image = _copra_on_paper((130, 130, 130), gradient=20)
    np.testing.assert_allclose(paper_color(image), (130, 130, 130), atol=1.5)


def test_normalized_paper_matches_reference_whatever_the_lighting():
    dim = normalize_lighting(_copra_on_paper((80, 86, 84)), 110)
    bright = normalize_lighting(_copra_on_paper((170, 178, 176)), 110)
    for image in (dim, bright):
        np.testing.assert_allclose(image[5, 5], (110, 110, 110), atol=2)


def test_same_scene_under_two_lights_normalizes_to_the_same_copra():
    # The copra block is lit by the same light as the paper: scale both.
    scene = _copra_on_paper((140, 140, 140)).astype(np.float32)
    dim = np.clip(scene * 0.7, 0, 255).astype(np.uint8)
    bright = np.clip(scene * 1.2, 0, 255).astype(np.uint8)
    a = normalize_lighting(dim, 110)[120, 160].astype(int)
    b = normalize_lighting(bright, 110)[120, 160].astype(int)
    assert np.abs(a - b).max() <= 3


def test_gains_clip_instead_of_wrapping():
    out = apply_gains(np.full((2, 2, 3), 200, np.uint8), np.array([2.0, 1.0, 0.5]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], (255, 200, 100))


def test_lighting_gains_are_reference_over_paper():
    gains = lighting_gains(_copra_on_paper((100, 110, 120)), 110)
    np.testing.assert_allclose(gains, (1.1, 1.0, 110 / 120), atol=0.02)


def test_rejects_grayscale():
    with pytest.raises(ValueError):
        normalize_lighting(np.zeros((10, 10), np.uint8), 110)


def _cfg(enabled):
    return {"preprocessing": {"masking_method": "background_grabcut", "resize_to": [64, 48],
                              "lighting_normalization": {"enabled": enabled, "reference_grey": 110}}}


def test_normalize_is_a_no_op_when_disabled():
    image = _copra_on_paper((150, 150, 150))
    assert normalize(image, _cfg(False)) is image
    assert normalize(image, {"preprocessing": {}}) is image


def test_preprocess_normalizes_the_copra():
    dim = _copra_on_paper((90, 90, 90))
    plain = preprocess(dim, _cfg(False))
    normed = preprocess(dim, _cfg(True))
    assert normed.shape == plain.shape
    copra = plain.any(axis=-1)
    assert normed[copra].mean() > plain[copra].mean()  # 110/90 gain brightens it
