import numpy as np
import pandas as pd

from copra_grading.cleaning import (
    flag_outliers,
    outlier_value_mask,
    remove_outliers,
)

CFG = {"cleaning": {"zscore_threshold": 3.0, "iqr_multiplier": 1.5}}


def _feature_frame(n_rows=40, n_feats=10):
    # Evenly spaced, bounded values so NO row is a natural IQR/z-score outlier.
    base = np.linspace(-1.0, 1.0, n_rows)
    data = {f"feat_{i}": base + 0.01 * i for i in range(n_feats)}
    df = pd.DataFrame(data)
    df.insert(0, "id", [f"P{r}" for r in range(n_rows)])
    df.insert(1, "batch_id", [f"B{r // 4}" for r in range(n_rows)])
    df.insert(2, "copra_class", "A")
    df.insert(3, "path", [f"photos/P{r}.jpg" for r in range(n_rows)])
    return df


def test_clean_frame_flags_nothing():
    assert flag_outliers(_feature_frame()).sum() == 0


def test_single_outlier_value_flags_its_row():
    df = _feature_frame()
    df.loc[df.index[5], "feat_0"] = 50.0  # one value out is enough

    flags = flag_outliers(df)

    assert bool(flags.loc[df.index[5]]) is True
    assert flags.drop(df.index[5]).sum() == 0


def test_iqr_rule_alone_flags_a_value():
    # Heavy tails inflate std, so a value past the IQR fence keeps |z| < 3.
    df = _feature_frame()
    col = np.linspace(-1.0, 1.0, len(df))
    col[[0, 1, 2, 3]] = [-10.0, -10.0, 10.0, 10.0]
    col[4] = 3.0
    df["feat_tailed"] = col

    mask = outlier_value_mask(df)

    z = abs(col[4] - col.mean()) / col.std()
    assert z < 3.0
    assert bool(mask.loc[df.index[4], "feat_tailed"]) is True


def test_zscore_rule_alone_flags_a_value():
    # IQR == 0 column (IQR rule inert) - only the z rule can catch the spike.
    df = _feature_frame()
    col = np.zeros(len(df))
    col[0] = 10.0
    df["feat_spike"] = col

    mask = outlier_value_mask(df)

    assert bool(mask.loc[df.index[0], "feat_spike"]) is True
    assert mask["feat_spike"].sum() == 1


def test_remove_outliers_drops_flagged_rows_only():
    df = _feature_frame()
    df.loc[df.index[[3, 9]], "feat_2"] = 80.0

    kept, flags = remove_outliers(df, CFG)

    assert len(kept) == len(df) - 2
    assert set(df.index[[3, 9]]).isdisjoint(kept.index)
    assert flags.sum() == 2
    assert not kept.isna().any().any()  # dropped, never imputed


def test_flags_series_is_index_aligned():
    df = _feature_frame()
    df.index = [f"row-{i}" for i in range(len(df))]

    flags = flag_outliers(df)

    assert list(flags.index) == list(df.index)
    assert flags.dtype == bool


def test_identifier_columns_never_contribute():
    df = _feature_frame(n_rows=12)
    df.loc[df.index[0], "copra_class"] = "F"
    df["batch_id"] = [f"B{i * 1000}" for i in range(12)]

    assert flag_outliers(df).sum() == 0


def test_zero_variance_feature_column_does_not_crash_or_flag():
    df = _feature_frame()
    df["feat_const"] = 7.0

    assert flag_outliers(df).sum() == 0


def test_all_non_numeric_frame_returns_all_false():
    df = pd.DataFrame({"batch_id": ["B1", "B1"], "note": ["a", "b"]})
    assert flag_outliers(df).tolist() == [False, False]
