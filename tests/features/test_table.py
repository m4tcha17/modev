import cv2
import numpy as np
import pandas as pd

from copra_grading.dataset import load_dataset
from copra_grading.features.table import FEATURE_EXCLUDE, build_feature_table

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


def test_one_row_per_photo_with_ids_and_finite_features(tmp_path):
    csv = _write_dataset(tmp_path)
    df = load_dataset(csv)

    table = build_feature_table(df, csv, CONFIG, n_jobs=1)

    assert len(table) == len(df)
    assert list(table["id"]) == list(df["id"])
    assert list(table.columns[:3]) == ["id", "batch_id", "copra_class"]
    feats = table.drop(columns=list(FEATURE_EXCLUDE))
    assert feats.shape[1] > 0
    assert np.isfinite(feats.to_numpy()).all()
    assert table["mask_fraction"].between(0.05, 0.5).all()
