import cv2
import numpy as np
import pandas as pd
import pytest

from copra_grading.augmentation import ORIGINAL, TRANSFORMS
from copra_grading.dataset import load_dataset
from copra_grading.features import EmptyRegionError, table as table_module
from copra_grading.features.table import (
    META_COLUMNS,
    build_feature_table,
    feature_columns,
    originals,
)

CONFIG = {
    "preprocessing": {"masking_method": "background_grabcut", "resize_to": [64, 48]},
    "features": {
        "glcm": {"distances": [1], "angles_deg": [0, 90]},
        "canny": {"low_threshold": 50, "high_threshold": 150},
    },
}


def _write_dataset(tmp_path, n_batches=2):
    (tmp_path / "photos").mkdir()
    rows = []
    for b in range(n_batches):
        for k in range(4):
            pid = f"p{b}{k}"
            img = np.full((120, 160, 3), 140, np.uint8)
            img[40:80, 50 + k * 5 : 110] = (230, 220, 200)
            img[40:55, 50 + k * 5 : 110] = (70, 40, 25)
            cv2.imwrite(str(tmp_path / "photos" / f"{pid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append({"id": pid, "batch_id": f"b{b}", "copra_class": "AB"[b], "path": f"photos/{pid}.png"})
    csv = tmp_path / "copra-dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_originals_only_gives_one_row_per_photo(tmp_path):
    csv = _write_dataset(tmp_path)
    df = load_dataset(csv)

    table = build_feature_table(df, csv, CONFIG, transforms=(ORIGINAL,), n_jobs=1)

    assert len(table) == len(df)
    assert list(table["id"]) == list(df["id"])
    assert list(table.columns[: len(META_COLUMNS)]) == list(META_COLUMNS)
    feats = table[feature_columns(table)]
    assert feats.shape[1] > 0
    assert np.isfinite(feats.to_numpy()).all()
    assert table["mask_fraction"].between(0.05, 0.5).all()


def test_all_transforms_gives_eight_rows_per_photo(tmp_path):
    csv = _write_dataset(tmp_path, n_batches=1)
    df = load_dataset(csv)

    table = build_feature_table(df, csv, CONFIG, n_jobs=1)

    assert len(table) == len(df) * len(TRANSFORMS)
    for _, group in table.groupby("id"):
        assert list(group["augment"]) == list(TRANSFORMS)
        assert group["mask_fraction"].nunique() == 1  # masked once per photo
    assert len(originals(table)) == len(df)


def test_rotation_swaps_glcm_directions(tmp_path):
    # A 90-degree turn turns horizontal texture into vertical texture, so
    # the 0- and 90-degree GLCM features trade places.
    csv = _write_dataset(tmp_path, n_batches=1)
    df = load_dataset(csv).head(1)

    # Square output size (as in configs/default.yaml): resizing then
    # commutes with a 90-degree turn, up to nearest-neighbour sampling.
    square = {**CONFIG, "preprocessing": {**CONFIG["preprocessing"], "resize_to": [64, 64]}}
    table = build_feature_table(df, csv, square, transforms=("rot0", "rot90"), n_jobs=1)
    r0, r90 = table.iloc[0], table.iloc[1]

    assert r90["glcm_d1_a90_contrast"] == pytest.approx(r0["glcm_d1_a0_contrast"], rel=0.05)
    assert r90["glcm_d1_a0_contrast"] == pytest.approx(r0["glcm_d1_a90_contrast"], rel=0.05)


def test_photo_with_no_copra_region_is_dropped_whole_and_reported(tmp_path, monkeypatch):
    csv = _write_dataset(tmp_path, n_batches=1)
    df = load_dataset(csv)
    real_extract = table_module.extract_all_features
    calls = {"n": 0}

    def flaky_extract(masked, config):
        # Fail on the 2nd photo's 3rd transform: every transform of that
        # photo must go, not just the failing one.
        calls["n"] += 1
        if calls["n"] == len(TRANSFORMS) + 3:
            raise EmptyRegionError("empty foreground mask: no copra pixels found")
        return real_extract(masked, config)

    monkeypatch.setattr(table_module, "extract_all_features", flaky_extract)
    with pytest.warns(UserWarning, match="1 of 4 photos dropped"):
        table = build_feature_table(df, csv, CONFIG, n_jobs=1)

    bad_id = df["id"].iloc[1]
    assert bad_id not in set(table["id"])
    assert len(table) == (len(df) - 1) * len(TRANSFORMS)
    assert [s["id"] for s in table.attrs["skipped"]] == [bad_id]


def test_other_value_errors_still_propagate(tmp_path, monkeypatch):
    # Only "no copra" is skippable; a bad dtype or config is a real bug.
    csv = _write_dataset(tmp_path, n_batches=1)
    df = load_dataset(csv)

    def broken_extract(masked, config):
        raise ValueError("expected uint8 image, got float32")

    monkeypatch.setattr(table_module, "extract_all_features", broken_extract)
    with pytest.raises(ValueError, match="expected uint8"):
        build_feature_table(df, csv, CONFIG, n_jobs=1)
