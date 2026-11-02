import numpy as np

from copra_grading.preprocessing.lighting import normalize_lighting, paper_color
from copra_grading.preprocessing.pipeline import mask_and_normalize, preprocess


def _scene(gain=1.0, h=300, w=400):
    """Grey paper with a top-to-bottom gradient, copra block in the middle.
    `gain` dims the whole scene, like the capture lights fading."""
    ramp = np.linspace(120, 160, h, dtype=np.float32)[:, None, None]
    image = np.broadcast_to(ramp * np.array([0.97, 1.0, 1.03]), (h, w, 3)).copy()
    mask = np.zeros((h, w), bool)
    mask[100:200, 120:280] = True
    image[100:150, 120:280] = (70, 40, 25)
    image[150:200, 120:280] = (235, 228, 210)
    return np.clip(image * gain, 0, 255).astype(np.uint8), mask


def test_paper_color_is_the_paper_under_the_copra():
    image, mask = _scene()
    # paper surface at rows 100-200 averages ramp value ~140 (G channel, gain 1.0)
    assert abs(paper_color(image, mask)[1] - 140) < 2
    # whole-frame fallback averages the full ramp, also ~140 here
    assert abs(paper_color(image)[1] - 140) < 2


def test_dimmed_photo_normalizes_back_to_same_copra_colors():
    ref = [130, 130, 130]
    bright, mask = _scene(1.0)
    dim, _ = _scene(0.75)
    out_bright = normalize_lighting(bright, paper_color(bright, mask), ref, mask)
    out_dim = normalize_lighting(dim, paper_color(dim, mask), ref, mask)
    diff = np.abs(out_bright[mask].astype(int) - out_dim[mask].astype(int))
    raw = np.abs(bright[mask].astype(int) - dim[mask].astype(int))
    assert diff.mean() < 2 < raw.mean()


def test_paper_maps_to_reference_grey():
    image, mask = _scene(0.8)
    out = normalize_lighting(image, paper_color(image, mask), [128, 128, 128], mask)
    # paper rows level with the copra, beside it
    assert np.allclose(out[100:200, :50].reshape(-1, 3).mean(axis=0), 128, atol=3)


def test_copra_pixels_stay_nonzero():
    image, mask = _scene()
    image[mask] = 1  # near-black copra, then a strong darkening gain
    out = normalize_lighting(image, np.array([250.0, 250, 250]), [60, 60, 60], mask)
    assert out[mask].any(axis=-1).all()


def test_pipeline_flag_turns_normalization_on_and_off():
    image, _ = _scene(0.7)
    base = {"masking_method": "background_grabcut", "resize_to": [64, 64],
            "reference_grey": [140, 140, 140]}
    off, _ = mask_and_normalize(image, {"preprocessing": {**base, "lighting_normalization": False}})
    on, mask = mask_and_normalize(image, {"preprocessing": {**base, "lighting_normalization": True}})
    assert (off == image).all()
    assert on[mask].mean() > image[mask].mean()  # dimmed scene brightened
    assert preprocess(image, {"preprocessing": {**base, "lighting_normalization": True}}).shape == (64, 64, 3)


def test_flat_field_removes_gradient_gain_does_not():
    from copra_grading.preprocessing.lighting import normalize
    image, mask = _scene()
    paper = ~mask
    paper[:, 100:300] = False  # side strips only
    g = normalize(image, mask, [128, 128, 128], "gain")[paper].astype(float)
    f = normalize(image, mask, [128, 128, 128], "flat_field")[paper].astype(float)
    assert f.std(axis=0).max() < 1.5 < g.std(axis=0).min()
