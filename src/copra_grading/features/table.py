"""Build the Step 3 feature table for a whole dataset. process.md Steps 1-3.

One row per photo: `id`, `batch_id`, `copra_class`, then every feature in
extract_all_features order, plus `mask_fraction` (share of the original
frame the mask covered) as a masking sanity column. `mask_fraction` is not
a model feature - drop it before training (see FEATURE_EXCLUDE).
"""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from copra_grading.dataset import load_image
from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import apply_mask, resize_masked
from copra_grading.preprocessing.pipeline import compute_mask

ID_COLUMNS = ("id", "batch_id", "copra_class")
FEATURE_EXCLUDE = ID_COLUMNS + ("mask_fraction",)


def _photo_row(csv_path: Path, row: dict, config: dict) -> dict:
    image = load_image(csv_path, row["path"])
    mask = compute_mask(image, config)
    masked = resize_masked(apply_mask(image, mask), config["preprocessing"]["resize_to"])
    out = {c: row[c] for c in ID_COLUMNS}
    out["mask_fraction"] = float(mask.mean())
    out.update(extract_all_features(masked, config))
    return out


def build_feature_table(
    df: pd.DataFrame, csv_path: Path, config: dict, n_jobs: int = -1
) -> pd.DataFrame:
    """Run masking + feature extraction on every row of `df` (as returned by
    dataset.load_dataset). Row order matches `df`.
    """
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_photo_row)(csv_path, record, config)
        for record in df.to_dict(orient="records")
    )
    return pd.DataFrame(rows, index=df.index)
