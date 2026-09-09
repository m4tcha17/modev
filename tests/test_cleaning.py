import numpy as np
import pandas as pd
import pytest

from copra_grading.cleaning import (
    DedupReport,
    DuplicateSampleError,
    SampleConflict,
    assert_clean,
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
