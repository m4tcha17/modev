"""GroupKFold splitting by Sample_ID. Spec §7a. Must run before augmentation - see ../../CLAUDE.md."""

import pandas as pd
from sklearn.model_selection import GroupKFold

# --- groupkfold ---


def split_by_sample(
    df: pd.DataFrame, n_splits: int, group_column: str = "Sample_ID"
) -> list[tuple[pd.Index, pd.Index]]:
    """Return (train_idx, val_idx) pairs, grouped so no Sample_ID crosses folds."""
    raise NotImplementedError
