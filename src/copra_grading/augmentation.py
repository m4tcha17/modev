"""Class-weighted geometric augmentation. process.md Step 4c. Multiplier: see ADR-002.

Training folds only, after the split. Geometric only - no photometric
augmentation, no SMOTE (see ../../CLAUDE.md). Smaller classes get more
augmented copies than larger ones.

Transforms are the 8 exact rotations/flips of a square grid (0/90/180/270
degrees, each with and without a horizontal flip). Arbitrary angles are
left out on purpose: they need pixel interpolation, which alters the
texture GLCM measures.

Augmented images go through Steps 2-3 like originals. Rotating/flipping the
masked crop is equivalent to rotating the photo and masking it again (the
background model is fitted from the whole border, so it has no preferred
orientation), so each photo is masked once and features are extracted for
every transform up front (features/table.py). A training fold then picks
which transformed rows to include - no re-extraction per fold or per
tuning trial.
"""

import zlib

import numpy as np
import pandas as pd

ORIGINAL = "rot0"
TRANSFORMS = (
    "rot0",
    "rot90",
    "rot180",
    "rot270",
    "flip",
    "flip_rot90",
    "flip_rot180",
    "flip_rot270",
)
MAX_EXTRA_COPIES = len(TRANSFORMS) - 1


def apply_transform(image: np.ndarray, name: str) -> np.ndarray:
    """Apply one of TRANSFORMS. Rotations are counter-clockwise; "flip" is
    a horizontal (left-right) flip applied before the rotation.
    """
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}")
    out = np.fliplr(image) if name.startswith("flip") else image
    quarter_turns = int(name.rsplit("rot", 1)[1]) // 90 if "rot" in name else 0
    return np.ascontiguousarray(np.rot90(out, k=quarter_turns))


def class_multipliers(
    class_counts: dict[str, int], base_copies: int, max_copies: int
) -> dict[str, int]:
    """Extra augmented copies per original image, per class.

    The largest class gets `base_copies`; a class with 1/k as many photos
    gets about k times as many images in total, capped at `max_copies`
    (and at MAX_EXTRA_COPIES, the number of distinct transforms).
    """
    if not class_counts:
        return {}
    cap = min(max_copies, MAX_EXTRA_COPIES)
    largest = max(class_counts.values())
    out = {}
    for cls, count in class_counts.items():
        total_per_image = (base_copies + 1) * largest / max(count, 1)
        out[cls] = int(min(cap, max(base_copies, round(total_per_image) - 1)))
    return out


def select_training_rows(
    table: pd.DataFrame,
    train_ids,
    multipliers: dict[str, int],
    seed: int = 0,
) -> pd.DataFrame:
    """Rows of a feature table (one row per photo per transform, column
    `augment`) for one training fold: every original in `train_ids`, plus
    `multipliers[class]` randomly chosen transformed copies of each.

    Choice of copies is reproducible from `seed` and the photo id.
    """
    train_ids = set(train_ids)
    rows = table[table["id"].isin(train_ids)]
    keep = []
    for photo_id, group in rows.groupby("id", sort=True):
        original = group[group["augment"] == ORIGINAL]
        keep.append(original)
        n_extra = multipliers.get(str(original["copra_class"].iloc[0]), 0)
        if n_extra <= 0:
            continue
        extras = group[group["augment"] != ORIGINAL]
        rng = np.random.default_rng([seed, zlib.crc32(str(photo_id).encode())])
        n_extra = min(n_extra, len(extras))
        picks = rng.choice(len(extras), size=n_extra, replace=False)
        keep.append(extras.iloc[np.sort(picks)])
    return pd.concat(keep) if keep else rows.iloc[0:0]
