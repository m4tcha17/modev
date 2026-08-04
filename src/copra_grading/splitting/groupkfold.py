"""GroupKFold split grouped by Sample_ID. Spec §7a.

All six angle images of one physical sample - and every augmented variant
later generated from them - must stay in the same fold. This is why splitting
happens before augmentation: an augmented copy of a training-fold sample must
never leak into a validation/test fold.
"""

import pandas as pd
from sklearn.model_selection import GroupKFold


def split_by_sample(
    df: pd.DataFrame, n_splits: int, group_column: str = "Sample_ID"
) -> list[tuple[pd.Index, pd.Index]]:
    """Return (train_idx, val_idx) pairs, grouped so no Sample_ID crosses folds."""
    raise NotImplementedError
