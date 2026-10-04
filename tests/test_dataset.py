import cv2
import numpy as np
import pandas as pd
import pytest

from copra_grading.dataset import (
    DatasetError,
    DatasetReport,
    assert_clean,
    check_batches,
    load_dataset,
    load_image,
)


def _batch(batch_id, copra_class, n_photos=4):
    ids = [f"{batch_id}-{i}" for i in range(n_photos)]
    return pd.DataFrame(
        {
            "id": ids,
            "batch_id": [batch_id] * n_photos,
            "copra_class": [copra_class] * n_photos,
            "path": [f"photos/{i}.jpg" for i in ids],
        }
    )


def _frame(*frames):
    return pd.concat(frames, ignore_index=True)


def test_load_dataset_reads_schema(tmp_path):
    csv = tmp_path / "copra-dataset.csv"
    _frame(_batch("b1", "A"), _batch("b2", "F")).to_csv(csv, index=False)

    df = load_dataset(csv)

    assert list(df.columns) == ["id", "batch_id", "copra_class", "path"]
    assert set(df["copra_class"]) == {"A", "F"}


def test_load_dataset_rejects_missing_column(tmp_path):
    csv = tmp_path / "copra-dataset.csv"
    _batch("b1", "A").drop(columns=["batch_id"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="batch_id"):
        load_dataset(csv)


def test_load_dataset_rejects_unknown_class(tmp_path):
    csv = tmp_path / "copra-dataset.csv"
    _batch("b1", "G").to_csv(csv, index=False)

    with pytest.raises(ValueError, match="G"):
        load_dataset(csv)


def test_load_image_resolves_path_against_csv_dir_and_returns_rgb(tmp_path):
    (tmp_path / "photos").mkdir()
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # pure red in BGR
    cv2.imwrite(str(tmp_path / "photos" / "x.png"), bgr)

    rgb = load_image(tmp_path / "copra-dataset.csv", "photos/x.png")

    assert rgb[0, 0].tolist() == [255, 0, 0]


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "copra-dataset.csv", "photos/nope.jpg")


def test_clean_frame_reports_no_issues():
    report = check_batches(_frame(_batch("b1", "A"), _batch("b2", "C")))

    assert isinstance(report, DatasetReport)
    assert report.is_clean is True


def test_short_batch_flagged_as_row_count():
    report = check_batches(_frame(_batch("b1", "A"), _batch("b2", "C", n_photos=3)))

    assert ("b2", "row_count") in {(i.batch_id, i.reason) for i in report.issues}


def test_mixed_class_batch_flagged():
    bad = _batch("b2", "C")
    bad.loc[bad.index[0], "copra_class"] = "D"

    report = check_batches(_frame(_batch("b1", "A"), bad))

    assert ("b2", "class_conflict") in {(i.batch_id, i.reason) for i in report.issues}


def test_duplicate_photo_id_flagged():
    bad = _batch("b2", "C")
    bad.loc[bad.index[-1], "id"] = bad.loc[bad.index[0], "id"]

    report = check_batches(_frame(_batch("b1", "A"), bad))

    assert ("b2", "duplicate_id") in {(i.batch_id, i.reason) for i in report.issues}


def test_issue_row_labels_point_into_source_frame():
    df = _frame(_batch("b1", "A"), _batch("b2", "C", n_photos=3))

    issue = next(i for i in check_batches(df).issues if i.batch_id == "b2")

    assert all(df.loc[label, "batch_id"] == "b2" for label in issue.row_labels)


def test_assert_clean_raises_on_dirty_frame():
    with pytest.raises(DatasetError) as excinfo:
        assert_clean(_frame(_batch("b1", "A"), _batch("b2", "C", n_photos=5)))

    assert "b2" in str(excinfo.value)


def test_assert_clean_returns_none_on_clean_frame():
    assert assert_clean(_frame(_batch("b1", "A"), _batch("b2", "C"))) is None


def test_class_metadata_covers_every_class():
    from copra_grading.dataset import CLASS_DESCRIPTIONS, MOISTURE_ORDER, VALID_CLASSES

    assert set(CLASS_DESCRIPTIONS) == set(VALID_CLASSES)
    assert sorted(MOISTURE_ORDER) == sorted(VALID_CLASSES)
    assert MOISTURE_ORDER[0] == "F" and MOISTURE_ORDER[-1] == "E"
