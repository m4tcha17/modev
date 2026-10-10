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

A photo with no usable copra region (EmptyRegionError in any transform) is
dropped whole - every transform of it - and listed in
`table.attrs["skipped"]`, never imputed.
"""

import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from copra_grading.augmentation import ORIGINAL, TRANSFORMS, apply_transform
from copra_grading.dataset import load_image
from copra_grading.features import EmptyRegionError
from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import apply_mask, resize_masked
from copra_grading.preprocessing.pipeline import compute_mask

ID_COLUMNS = ("id", "batch_id", "copra_class")
META_COLUMNS = ID_COLUMNS + ("augment", "mask_fraction")


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model feature columns of a feature table, in extraction order."""
    return [c for c in table.columns if c not in META_COLUMNS]


def _photo_rows(
    csv_path: Path, row: dict, config: dict, transforms
) -> tuple[list[dict], dict | None]:
    """(feature rows, None), or ([], skip record) if the photo has no usable copra."""
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
        try:
            record.update(extract_all_features(masked, config))
        except EmptyRegionError as err:
            return [], {"id": row["id"], "augment": name, "reason": str(err)}
        out.append(record)
    return out, None


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
    table = pd.DataFrame([r for rows, _ in per_photo for r in rows])
    skipped = [skip for _, skip in per_photo if skip is not None]
    table.attrs["skipped"] = skipped
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {len(df)} photos dropped, no usable copra region: "
            + ", ".join(s["id"] for s in skipped),
            stacklevel=2,
        )
    return table


def originals(table: pd.DataFrame) -> pd.DataFrame:
    """Only the un-augmented rows (one per photo)."""
    return table[table["augment"] == ORIGINAL]
