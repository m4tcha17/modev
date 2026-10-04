import numpy as np

from copra_grading.preprocessing.otsu import apply_mask, mask_image


def test_mask_image_isolates_bright_square_on_dark_background():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[15:25, 15:25] = 255

    mask = mask_image(image)

    assert mask.shape == (40, 40)
    assert mask.dtype == bool
    assert mask[20, 20]
    assert not mask[0, 0]
    assert mask.sum() == 100


def test_mask_image_isolates_dark_square_on_bright_background():
    image = np.full((40, 40), 200, dtype=np.uint8)
    image[15:25, 15:25] = 10

    mask = mask_image(image)

    assert mask[20, 20]
    assert not mask[0, 0]
    assert mask.sum() == 100


def test_apply_mask_crops_to_bounding_box_and_zeroes_background():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    # Cross shape: the bounding box's own corners fall outside the shape,
    # so the crop must still zero those corner pixels.
    mask[8:12, 2:18] = True
    mask[2:18, 8:12] = True
    image[mask] = [100, 150, 200]

    masked_image = apply_mask(image, mask)

    assert masked_image.shape == (16, 16, 3)
    assert list(masked_image[7, 7]) == [100, 150, 200]
    assert list(masked_image[0, 0]) == [0, 0, 0]


def test_mask_image_selects_centered_object_covering_more_than_half_the_frame():
    # 34x34 bright square centered in a 40x40 dark frame: 1156/1600 = 72.25%
    # of the frame - the object is now the *majority*-count side, which is
    # exactly the case the old "smaller side wins" heuristic got backwards.
    image = np.zeros((40, 40), dtype=np.uint8)
    image[3:37, 3:37] = 255

    mask = mask_image(image)

    assert mask[20, 20]        # frame center, inside the object
    assert not mask[0, 0]      # corner, inside the background
    assert mask.sum() == 34 * 34


def test_apply_mask_uint8_mask_matches_bool_mask():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    bool_mask = np.zeros((20, 20), dtype=bool)
    bool_mask[8:12, 2:18] = True
    bool_mask[2:18, 8:12] = True
    image[bool_mask] = [100, 150, 200]

    uint8_mask = bool_mask.astype(np.uint8) * 255

    result_from_bool = apply_mask(image, bool_mask)
    result_from_uint8 = apply_mask(image, uint8_mask)

    np.testing.assert_array_equal(result_from_bool, result_from_uint8)


def test_resize_masked_hits_target_size_and_keeps_background_zero():
    from copra_grading.preprocessing.otsu import resize_masked

    img = np.zeros((30, 40, 3), dtype=np.uint8)
    img[10:20, 10:30] = 180

    out = resize_masked(img, (64, 64))

    assert out.shape == (64, 64, 3)
    # nearest-neighbour: only original values survive, no blended edge pixels
    assert set(np.unique(out)) <= {0, 180}


def test_apply_mask_keeps_pure_black_copra_pixels_nonzero():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True  # copra region is pure black

    out = apply_mask(image, mask)

    assert np.all(np.any(out != 0, axis=-1))
