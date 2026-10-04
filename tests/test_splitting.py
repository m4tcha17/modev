import numpy as np
import pandas as pd
import pytest

from copra_grading.splitting import split_by_batch

CLASSES = ("A", "B", "C", "D", "E", "F")


def _frame(n_batches=30, n_photos=4):
    rows = []
    for b in range(n_batches):
        for p in range(n_photos):
            rows.append(
                {
                    "id": f"B{b}-{p}",
                    "batch_id": f"B{b}",
                    "copra_class": CLASSES[b % len(CLASSES)],
                    "feat": b + p,
                }
            )
    return pd.DataFrame(rows)


def test_returns_n_splits_pairs():
    folds = split_by_batch(_frame(), n_splits=5)

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)


def test_no_batch_id_crosses_a_fold():
    df = _frame()
    for train_idx, val_idx in split_by_batch(df, n_splits=5):
        train_ids = set(df.iloc[train_idx]["batch_id"])
        val_ids = set(df.iloc[val_idx]["batch_id"])
        assert train_ids.isdisjoint(val_ids)


def test_every_row_in_exactly_one_validation_fold():
    df = _frame()
    seen = np.zeros(len(df), dtype=int)
    for _, val_idx in split_by_batch(df, n_splits=5):
        seen[val_idx] += 1

    assert (seen == 1).all()


def test_folds_are_stratified_by_class():
    # 30 batches, 5 per class, 5 folds -> each validation fold gets one
    # batch of every class.
    df = _frame()
    for _, val_idx in split_by_batch(df, n_splits=5):
        assert set(df.iloc[val_idx]["copra_class"]) == set(CLASSES)


def test_same_random_state_gives_same_folds():
    df = _frame()
    a = split_by_batch(df, n_splits=5, random_state=7)
    b = split_by_batch(df, n_splits=5, random_state=7)

    for (ta, va), (tb, vb) in zip(a, b):
        assert np.array_equal(ta, tb) and np.array_equal(va, vb)


def test_more_splits_than_groups_raises():
    with pytest.raises(ValueError):
        split_by_batch(_frame(n_batches=3), n_splits=5)


def test_missing_group_column_raises():
    with pytest.raises(ValueError):
        split_by_batch(_frame().drop(columns=["batch_id"]), n_splits=3)


def test_missing_label_column_raises():
    with pytest.raises(ValueError):
        split_by_batch(_frame().drop(columns=["copra_class"]), n_splits=3)


def test_custom_group_column_keeps_whole_samples_together():
    # Two batches per whole copra sample: grouping by sample_id must keep
    # both batches of a sample in the same fold.
    df = _frame()
    df["sample_id"] = df["batch_id"].str[1:].astype(int).floordiv(2).map(lambda s: f"S{s}")
    df["copra_class"] = df["sample_id"].map(
        lambda s: CLASSES[int(s[1:]) % len(CLASSES)]
    )

    for train_idx, val_idx in split_by_batch(df, n_splits=3, group_column="sample_id"):
        train_ids = set(df.iloc[train_idx]["sample_id"])
        val_ids = set(df.iloc[val_idx]["sample_id"])
        assert train_ids.isdisjoint(val_ids)
