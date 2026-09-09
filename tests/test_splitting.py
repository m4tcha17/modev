import numpy as np
import pandas as pd
import pytest

from copra_grading.splitting import split_by_sample


def _frame(n_samples=10, n_angles=6):
    rows = []
    for s in range(n_samples):
        for a in range(n_angles):
            rows.append({"Sample_ID": f"S{s}", "Angle_ID": a + 1, "feat": s + a})
    return pd.DataFrame(rows)


def test_returns_n_splits_pairs():
    folds = split_by_sample(_frame(), n_splits=5)

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)


def test_no_sample_id_crosses_a_fold():
    df = _frame()
    for train_idx, val_idx in split_by_sample(df, n_splits=5):
        train_ids = set(df.iloc[train_idx]["Sample_ID"])
        val_ids = set(df.iloc[val_idx]["Sample_ID"])
        assert train_ids.isdisjoint(val_ids)


def test_every_row_in_exactly_one_validation_fold():
    df = _frame()
    seen = np.zeros(len(df), dtype=int)
    for _, val_idx in split_by_sample(df, n_splits=5):
        seen[val_idx] += 1

    assert (seen == 1).all()


def test_more_splits_than_groups_raises():
    df = _frame(n_samples=3)
    with pytest.raises(ValueError):
        split_by_sample(df, n_splits=5)


def test_missing_group_column_raises():
    df = _frame().drop(columns=["Sample_ID"])
    with pytest.raises(ValueError):
        split_by_sample(df, n_splits=3)


def test_custom_group_column():
    df = _frame().rename(columns={"Sample_ID": "batch"})
    folds = split_by_sample(df, n_splits=5, group_column="batch")

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_ids = set(df.iloc[train_idx]["batch"])
        val_ids = set(df.iloc[val_idx]["batch"])
        assert train_ids.isdisjoint(val_ids)
