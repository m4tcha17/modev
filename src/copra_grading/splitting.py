"""StratifiedGroupKFold splitting. process.md Step 4b. Must run before augmentation - see ../../CLAUDE.md.

Grouping keeps every photo of one group in the same fold; stratification
keeps each fold's class (A-F) mix close to the whole dataset's.

Grouping must be at the level of the whole physical copra sample. Several
batch_ids can come from the same whole sample, so grouping by batch_id
alone can still put one sample in both training and validation. Point
group_column at a whole-sample ID column once the CSV has one.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --- stratified_groupkfold ---


def split_by_batch(
    df: pd.DataFrame,
    n_splits: int,
    group_column: str = "batch_id",
    label_column: str = "copra_class",
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold split, grouped by group_column, stratified by
    label_column, run BEFORE augmentation.

    Every photo in one group - and every augmented variant generated from
    them later - stays in the same fold. Returns n_splits (train_idx,
    val_idx) pairs of positional row indices. Shuffled with a fixed
    random_state so folds are reproducible.
    """
    for column in (group_column, label_column):
        if column not in df.columns:
            raise ValueError(f"column {column!r} not in dataframe columns")

    groups = df[group_column].to_numpy()
    n_groups = pd.unique(groups).size
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of distinct "
            f"{group_column} groups ({n_groups})"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return [
        (train_idx, val_idx)
        for train_idx, val_idx in splitter.split(
            df, y=df[label_column].to_numpy(), groups=groups
        )
    ]
