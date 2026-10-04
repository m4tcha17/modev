"""Build the Step 3 feature table for a whole dataset. process.md Steps 1-3 (+ 4c cache).

One row per photo per augmentation transform: `id`, `batch_id`,
`copra_class`, `augment` (transform name, "rot0" = the original photo),
`mask_fraction` (share of the original frame the mask covered - a masking
sanity column, not a model feature), then every feature in
extract_all_features order.

Each photo is masked once; the masked crop is rotated/flipped and features
are extracted per transform (see augmentation.py for why that is
equivalent to augmenting the photo). Which transformed rows a training
fold actually uses is decided later, by augmentation.select_training_rows.
"""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from copra_grading.augmentation import ORIGINAL, TRANSFORMS, apply_transform
from copra_grading.dataset import load_image
from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import apply_mask, resize_masked
from copra_grading.preprocessing.pipeline import compute_mask

ID_COLUMNS = ("id", "batch_id", "copra_class")
META_COLUMNS = ID_COLUMNS + ("augment", "mask_fraction")


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model feature columns of a feature table, in extraction order."""
    return [c for c in table.columns if c not in META_COLUMNS]


def _photo_rows(csv_path: Path, row: dict, config: dict, transforms) -> list[dict]:
    image = load_image(csv_path, row["path"])
    mask = compute_mask(image, config)
    crop = apply_mask(image, mask)
    size = config["preprocessing"]["resize_to"]
    out = []
    for name in transforms:
        masked = resize_masked(apply_transform(crop, name), size)
        record = {c: row[c] for c in ID_COLUMNS}
        record["augment"] = name
        record["mask_fraction"] = float(mask.mean())
        record.update(extract_all_features(masked, config))
        out.append(record)
    return out


def build_feature_table(
    df: pd.DataFrame,
    csv_path: Path,
    config: dict,
    transforms=TRANSFORMS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run masking + feature extraction on every row of `df` (as returned by
    dataset.load_dataset), for every transform in `transforms`. Pass
    `transforms=(ORIGINAL,)` to skip augmentation (e.g. at inference time).
    Photos keep `df` order; transforms follow `transforms` order.
    """
    per_photo = Parallel(n_jobs=n_jobs)(
        delayed(_photo_rows)(csv_path, record, config, transforms)
        for record in df.to_dict(orient="records")
    )
    return pd.DataFrame([r for rows in per_photo for r in rows])


def originals(table: pd.DataFrame) -> pd.DataFrame:
    """Only the un-augmented rows (one per photo)."""
    return table[table["augment"] == ORIGINAL]
