import numpy as np
import pytest

from copra_grading.preprocessing.background import mask_image_background
from copra_grading.preprocessing.otsu import mask_image
from copra_grading.preprocessing.pipeline import compute_mask, preprocess


def _two_tone_on_grey(h=300, w=400):
    """Mimics the real photos: grey background whose brightness drifts top to
    bottom, copra with a dark brown skin half and a bright white meat half.
    The background's lightness sits between the two copra tones.
    """
    ramp = np.linspace(110, 165, h, dtype=np.float32)[:, None]
    image = np.empty((h, w, 3), np.float32)
    image[..., 0] = ramp - 5
    image[..., 1] = ramp + 3
    image[..., 2] = ramp + 5  # slightly blue-grey, like the real tray
    truth = np.zeros((h, w), bool)
    truth[100:200, 120:280] = True
    image[100:140, 120:280] = (70, 40, 25)     # brown skin
    image[140:200, 120:280] = (235, 228, 210)  # white meat
    rng = np.random.default_rng(0)
    image += rng.normal(0, 2, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), truth


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_otsu_fails_on_two_tone_copra_on_grey():
    # Regression record of why Otsu was replaced.
    # One global threshold can't keep both tones apart from a background
    # lying between them: here the brown skin is lost.
    image, truth = _two_tone_on_grey()
    mask = mask_image(image)
    assert not mask[120, 200]
    assert _iou(mask, truth) < 0.7


def test_background_mask_recovers_both_tones():
    image, truth = _two_tone_on_grey()

    mask = mask_image_background(image)

    assert mask.shape == truth.shape and mask.dtype == bool
    assert _iou(mask, truth) > 0.9
    assert mask[120, 200] and mask[170, 200]  # skin and meat both kept
    assert not mask[10, 10]


def test_background_mask_without_grabcut_still_finds_object():
    image, truth = _two_tone_on_grey()
    assert _iou(mask_image_background(image, grabcut_iters=0), truth) > 0.85


def test_blank_background_gives_empty_mask():
    image = np.full((200, 300, 3), 140, np.uint8)
    assert not mask_image_background(image).any()


def test_rejects_grayscale_input():
    with pytest.raises(ValueError):
        mask_image_background(np.zeros((50, 50), np.uint8))


def _cfg(method):
    return {"preprocessing": {"masking_method": method, "resize_to": [64, 48]}}


def test_compute_mask_dispatches_on_method():
    image, truth = _two_tone_on_grey()
    assert _iou(compute_mask(image, _cfg("background_grabcut")), truth) > 0.9
    np.testing.assert_array_equal(compute_mask(image, _cfg("otsu")), mask_image(image))


def test_compute_mask_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_mask(np.zeros((10, 10, 3), np.uint8), _cfg("magic"))


def test_preprocess_returns_fixed_size_masked_image():
    image, _ = _two_tone_on_grey()

    out = preprocess(image, _cfg("background_grabcut"))

    assert out.shape == (48, 64, 3)
    assert out.any()
