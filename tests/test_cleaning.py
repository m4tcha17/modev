import numpy as np
import pandas as pd
import pytest

from copra_grading.cleaning import (
    DedupReport,
    DuplicateSampleError,
    SampleConflict,
    assert_clean,
    flag_outliers,
    flag_outliers_iqr,
    flag_outliers_zscore,
    resolve_duplicate_samples,
)


def _sample_rows(sample_id, moisture, n_angles=6):
    return pd.DataFrame(
        {
            "Sample_ID": [sample_id] * n_angles,
            "Angle_ID": list(range(1, n_angles + 1)),
            "moisture_reading": [moisture] * n_angles,
            "feat_a": np.linspace(0.0, 1.0, n_angles),
        }
    )


def _frame(*frames):
    return pd.concat(frames, ignore_index=True)


def test_clean_frame_reports_no_conflicts():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0))

    report = resolve_duplicate_samples(df)

    assert isinstance(report, DedupReport)
    assert report.is_clean is True
    assert report.conflicts == []


def test_short_sample_flagged_as_row_count_conflict():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    report = resolve_duplicate_samples(df)

    assert report.is_clean is False
    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "row_count") in reasons


def test_duplicate_angle_flagged():
    bad = _sample_rows("S2", 20.0)
    bad.loc[bad.index[-1], "Angle_ID"] = 1  # angle 1 now appears twice
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "duplicate_angle") in reasons


def test_conflicting_moisture_reading_flagged():
    bad = _sample_rows("S2", 20.0)
    bad.loc[bad.index[0], "moisture_reading"] = 4.0  # two physical samples, one ID
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "moisture_conflict") in reasons


def test_multiple_issues_produce_one_conflict_each():
    bad = _sample_rows("S2", 20.0, n_angles=5)
    bad.loc[bad.index[0], "moisture_reading"] = 4.0
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = sorted(c.reason for c in report.conflicts if c.sample_id == "S2")
    assert reasons == ["moisture_conflict", "row_count"]


def test_conflict_carries_row_labels_into_source_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    report = resolve_duplicate_samples(df)
    conflict = next(c for c in report.conflicts if c.sample_id == "S2")

    assert set(conflict.row_labels).issubset(set(df.index))
    assert all(df.loc[label, "Sample_ID"] == "S2" for label in conflict.row_labels)


def test_assert_clean_raises_on_dirty_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    with pytest.raises(DuplicateSampleError) as excinfo:
        assert_clean(df)

    assert isinstance(excinfo.value.report, DedupReport)
    assert "S2" in str(excinfo.value)


def test_assert_clean_returns_none_on_clean_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0))

    assert assert_clean(df) is None


def _feature_frame(n_rows=40, n_feats=10, seed=0):
    # Fixture tightened (controller pre-authorized): explicit bounded, evenly
    # spaced values so NO row is a natural IQR/Z-score outlier. This keeps the
    # intent of the min_flagged_features=1 tests unambiguous. IQR and z-score
    # are scale-invariant, so shrinking a random spread would not have helped.
    base = np.linspace(-1.0, 1.0, n_rows)
    data = {f"feat_{i}": base + 0.01 * i for i in range(n_feats)}
    df = pd.DataFrame(data)
    df.insert(0, "Sample_ID", [f"S{r // 6}" for r in range(n_rows)])
    df.insert(1, "Angle_ID", [r % 6 + 1 for r in range(n_rows)])
    df.insert(2, "moisture_reading", 8.0)
    return df


def test_iqr_flags_a_row_with_many_extreme_features():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[3], feat_cols] = 50.0  # every feature wildly out

    flags = flag_outliers_iqr(df, multiplier=1.5, min_flagged_features=3)

    assert flags.loc[df.index[3]] is np.True_ or bool(flags.loc[df.index[3]]) is True
    assert flags.drop(df.index[3]).sum() == 0


def test_min_flagged_features_threshold_respected():
    df = _feature_frame()
    df.loc[df.index[5], ["feat_0", "feat_1"]] = 50.0  # exactly 2 features out

    assert bool(flag_outliers_iqr(df, min_flagged_features=3).loc[df.index[5]]) is False
    assert bool(flag_outliers_iqr(df, min_flagged_features=2).loc[df.index[5]]) is True


def test_flags_series_is_index_aligned():
    df = _feature_frame()
    df.index = [f"row-{i}" for i in range(len(df))]

    flags = flag_outliers_iqr(df)

    assert list(flags.index) == list(df.index)
    assert flags.dtype == bool


def test_zscore_also_flags_the_obvious_outlier():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[7], feat_cols] = 40.0

    flags = flag_outliers_zscore(df, threshold=3.0, min_flagged_features=3)

    assert bool(flags.loc[df.index[7]]) is True


def test_identifier_columns_never_contribute():
    df = _feature_frame(n_rows=12)
    df["Angle_ID"] = df["Angle_ID"] * 1000  # large, but must be ignored
    df["moisture_reading"] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 99.0, 4.0, 4.0, 4.0, 4.0, 4.0]

    flags = flag_outliers_iqr(df, min_flagged_features=1)

    assert flags.sum() == 0


def test_zero_variance_feature_column_does_not_crash_or_flag():
    df = _feature_frame()
    df["feat_const"] = 7.0

    flags = flag_outliers_iqr(df, min_flagged_features=1)

    assert flags.sum() == 0


def test_iqr_ignores_degenerate_column_with_mostly_identical_values():
    # Non-constant column where Q1 == Q3 == 0 -> IQR == 0. The 1.5x multiplier
    # is inert, so without a zero-IQR guard every nonzero value reads as an
    # outlier (spec §6.2 step 2: a zero-variance column contributes no flags).
    df = _feature_frame()
    col = np.zeros(len(df))
    col[[1, 2]] = 1e-9  # 38 zeros + 2 tiny nonzero -> Q1==Q3==0, not constant
    df["feat_degenerate"] = col

    flags = flag_outliers_iqr(df, min_flagged_features=1)

    assert flags.sum() == 0


def test_flag_outliers_dispatches_on_config_method():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[2], feat_cols] = 60.0

    cfg = {
        "cleaning": {
            "outlier_method": "zscore",
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3.0,
            "outlier_min_features": 3,
        }
    }

    flags = flag_outliers(df, cfg)
    assert bool(flags.loc[df.index[2]]) is True


def test_flag_outliers_rejects_unknown_method():
    cfg = {"cleaning": {"outlier_method": "madness", "outlier_min_features": 3}}
    with pytest.raises(ValueError):
        flag_outliers(_feature_frame(), cfg)


def test_all_non_numeric_frame_returns_all_false():
    df = pd.DataFrame({"Sample_ID": ["S1", "S1"], "note": ["a", "b"]})
    flags = flag_outliers_iqr(df)
    assert flags.tolist() == [False, False]
