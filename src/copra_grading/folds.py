"""Cross-validation folds from a cleaned feature table. process.md Steps 4b-4c.

Split first, augment second (see ../../CLAUDE.md): StratifiedGroupKFold runs
on the original rows only, then each training fold picks its augmented copies
from the photos already assigned to it. Validation folds are originals only.
Multipliers come from each training fold's own class counts (ADR-002).
"""

from dataclasses import dataclass

import pandas as pd

from copra_grading.augmentation import ORIGINAL, class_multipliers, select_training_rows
from copra_grading.splitting import split_by_batch


@dataclass(frozen=True)
class Fold:
    index: int
    train: pd.DataFrame  # originals + augmented copies, training photos only
    val: pd.DataFrame  # originals only
    multipliers: dict[str, int]


def make_folds(table: pd.DataFrame, config: dict) -> list[Fold]:
    """Split the original rows, then build each fold's training rows.

    No photo id - and so no group, since a group's photos share a fold -
    appears in both a fold's train and val.
    """
    split_cfg = config["splitting"]
    aug_cfg = config["augmentation"]
    label = split_cfg["label_column"]

    originals = table[table["augment"] == ORIGINAL].reset_index(drop=True)
    splits = split_by_batch(
        originals,
        n_splits=split_cfg["n_splits"],
        group_column=split_cfg["group_column"],
        label_column=label,
        random_state=split_cfg["random_state"],
    )

    folds = []
    for k, (train_idx, val_idx) in enumerate(splits):
        train_originals = originals.iloc[train_idx]
        counts = {str(c): int(n) for c, n in train_originals[label].value_counts().items()}
        multipliers = class_multipliers(
            counts, aug_cfg["base_copies"], aug_cfg["max_multiplier"]
        )
        train = select_training_rows(
            table,
            train_originals["id"],
            multipliers,
            seed=split_cfg["random_state"] + k,
        )
        folds.append(Fold(k, train, originals.iloc[val_idx], multipliers))
    return folds
