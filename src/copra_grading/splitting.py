"""GroupKFold splitting by Sample_ID. Spec §7a. Must run before augmentation - see ../../CLAUDE.md."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# --- groupkfold ---


def split_by_sample(
    df: pd.DataFrame, n_splits: int, group_column: str = "Sample_ID"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold split, grouped by group_column, run BEFORE augmentation.

    Every angle image of one physical sample - and every augmented variant
    generated from it later - stays in the same fold. Returns n_splits
    (train_idx, val_idx) pairs of positional row indices.
    """
    if group_column not in df.columns:
        raise ValueError(f"group_column {group_column!r} not in dataframe columns")

    groups = df[group_column].to_numpy()
    n_groups = pd.unique(groups).size
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of distinct "
            f"{group_column} groups ({n_groups})"
        )

    splitter = GroupKFold(n_splits=n_splits)
    return [
        (train_idx, val_idx)
        for train_idx, val_idx in splitter.split(df, groups=groups)
    ]
