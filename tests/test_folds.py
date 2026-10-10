import pandas as pd

from copra_grading.augmentation import TRANSFORMS
from copra_grading.folds import make_folds

CLASSES = ("A", "B", "C", "D", "E", "F")
CONFIG = {
    "splitting": {
        "n_splits": 5,
        "group_column": "batch_id",
        "label_column": "copra_class",
        "random_state": 42,
    },
    "augmentation": {"base_copies": 1, "max_multiplier": 4},
}


def _table(n_batches=30, n_photos=4):
    rows = []
    for b in range(n_batches):
        # Class A three times as common as the rest, so multipliers differ.
        cls = "A" if b % 2 == 0 else CLASSES[1 + (b // 2) % 5]
        for p in range(n_photos):
            for t in TRANSFORMS:
                rows.append(
                    {"id": f"B{b}-{p}", "batch_id": f"B{b}", "copra_class": cls,
                     "augment": t, "mask_fraction": 0.5, "feat": float(b + p)}
                )
    return pd.DataFrame(rows)


def test_every_photo_validated_exactly_once():
    table = _table()
    folds = make_folds(table, CONFIG)

    val_ids = pd.concat([f.val["id"] for f in folds])
    assert len(folds) == 5
    assert not val_ids.duplicated().any()
    assert set(val_ids) == set(table["id"])


def test_no_photo_or_group_in_both_train_and_val():
    for fold in make_folds(_table(), CONFIG):
        assert set(fold.train["id"]).isdisjoint(fold.val["id"])
        assert set(fold.train["batch_id"]).isdisjoint(fold.val["batch_id"])


def test_val_is_originals_only_train_has_augmented_copies():
    for fold in make_folds(_table(), CONFIG):
        assert (fold.val["augment"] == "rot0").all()
        assert (fold.train["augment"] != "rot0").any()
        # every training photo keeps its original row
        train_originals = fold.train[fold.train["augment"] == "rot0"]
        assert set(train_originals["id"]) == set(fold.train["id"])


def test_smaller_classes_get_more_copies_per_fold():
    for fold in make_folds(_table(), CONFIG):
        m = fold.multipliers
        assert m["A"] == 1
        assert all(m[c] > m["A"] for c in CLASSES[1:])
        assert max(m.values()) <= 4
