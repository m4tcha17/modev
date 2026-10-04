import numpy as np
import pandas as pd
import pytest

from copra_grading.augmentation import (
    MAX_EXTRA_COPIES,
    ORIGINAL,
    TRANSFORMS,
    apply_transform,
    class_multipliers,
    select_training_rows,
)


def test_eight_distinct_transforms_of_an_asymmetric_image():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    outs = [apply_transform(img, t) for t in TRANSFORMS]

    assert len({o.tobytes() + bytes(o.shape) for o in outs}) == 8
    np.testing.assert_array_equal(outs[0], img)


def test_rot90_and_flip_match_numpy():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    np.testing.assert_array_equal(apply_transform(img, "rot90"), np.rot90(img))
    np.testing.assert_array_equal(apply_transform(img, "flip"), np.fliplr(img))
    np.testing.assert_array_equal(apply_transform(img, "flip_rot180"), np.rot90(np.fliplr(img), 2))


def test_transform_keeps_rgb_channels():
    img = np.zeros((3, 5, 3), np.uint8)
    assert apply_transform(img, "rot270").shape == (5, 3, 3)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        apply_transform(np.zeros((2, 2)), "rot45")


def test_smaller_class_gets_more_copies():
    m = class_multipliers({"A": 200, "B": 100}, base_copies=1, max_copies=7)
    assert m["A"] == 1
    assert m["B"] == 3  # half as many photos -> twice as many images in total


def test_multiplier_is_capped():
    m = class_multipliers({"A": 1000, "B": 10}, base_copies=1, max_copies=4)
    assert m["B"] == 4
    assert class_multipliers({"A": 1000, "B": 1}, 1, 99)["B"] == MAX_EXTRA_COPIES


def test_balanced_classes_get_base_copies():
    assert class_multipliers({"A": 50, "B": 50}, base_copies=2, max_copies=7) == {"A": 2, "B": 2}


def _table():
    rows = []
    for pid, cls in (("p1", "A"), ("p2", "A"), ("p3", "B")):
        for t in TRANSFORMS:
            rows.append({"id": pid, "batch_id": pid, "copra_class": cls, "augment": t, "f": 1.0})
    return pd.DataFrame(rows)


def test_select_training_rows_keeps_originals_and_adds_copies():
    out = select_training_rows(_table(), ["p1", "p3"], {"A": 1, "B": 3}, seed=0)

    assert set(out["id"]) == {"p1", "p3"}  # p2 is not in the training fold
    assert (out[out["augment"] == ORIGINAL]["id"].sort_values().tolist()) == ["p1", "p3"]
    assert (out["id"] == "p1").sum() == 2
    assert (out["id"] == "p3").sum() == 4
    assert not out.duplicated(["id", "augment"]).any()


def test_select_training_rows_is_reproducible():
    a = select_training_rows(_table(), ["p1", "p2", "p3"], {"A": 2, "B": 2}, seed=5)
    b = select_training_rows(_table(), ["p1", "p2", "p3"], {"A": 2, "B": 2}, seed=5)
    pd.testing.assert_frame_equal(a, b)
