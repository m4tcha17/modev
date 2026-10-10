"""Feature-level outlier removal. process.md Step 4a.

Runs on extracted feature values, not images. A single value is an outlier
if |z| > zscore_threshold OR it falls outside
[Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR]. Any photo row holding at
least one outlier value is dropped - no imputation, and the thresholds are
never relaxed. If too much data is lost, collect more data.
"""

import numpy as np
import pandas as pd

from copra_grading.augmentation import ORIGINAL

# Never treated as features. `augment` and `mask_fraction` come from the
# feature table (features/table.py META_COLUMNS); mask_fraction is numeric,
# so without this it would be outlier-checked like a feature.
_IDENTIFIER_COLUMNS = ("id", "batch_id", "copra_class", "path", "augment", "mask_fraction")


def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    kept = [c for c in features.columns if c not in _IDENTIFIER_COLUMNS]
    return features[kept].select_dtypes(include="number")


def outlier_value_mask(
    features: pd.DataFrame, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """Boolean frame, True where a numeric feature value is an outlier by
    either rule. Zero-variance columns (std == 0 or IQR == 0) contribute no
    flags under the rule that degenerates.
    """
    numeric = _numeric_features(features)
    if numeric.empty:
        return numeric.astype(bool)

    std = numeric.std(ddof=0).replace(0.0, np.nan)
    z = (numeric - numeric.mean()).abs().div(std, axis=1)
    z_flags = z.gt(zscore_threshold).fillna(False)

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    iqr_flags = numeric.lt(q1 - iqr_multiplier * iqr, axis=1) | numeric.gt(
        q3 + iqr_multiplier * iqr, axis=1
    )
    # IQR == 0 makes the fence collapse onto Q1, so every value != Q1 would
    # read as an outlier - such columns contribute no IQR flags.
    iqr_flags.loc[:, iqr <= 0] = False

    return z_flags | iqr_flags


def flag_outliers(
    features: pd.DataFrame, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5
) -> pd.Series:
    """True for every row with at least one outlier value. Index-aligned."""
    mask = outlier_value_mask(features, zscore_threshold, iqr_multiplier)
    return mask.any(axis=1).reindex(features.index, fill_value=False).astype(bool)


def remove_outliers(features: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Drop flagged rows. Return (kept rows, the boolean flag Series) so the
    caller can report how much data was lost.
    """
    cleaning_cfg = config["cleaning"]
    flags = flag_outliers(
        features, cleaning_cfg["zscore_threshold"], cleaning_cfg["iqr_multiplier"]
    )
    return features.loc[~flags], flags


def remove_outlier_photos(
    table: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Outlier removal for a feature table with augmented rows (column
    `augment`, one row per photo per transform).

    Flags are computed on the original rows only - the rotated/flipped copies
    would otherwise skew the mean, std and quartiles. A flagged photo is
    dropped with every one of its transformed rows. Returns (kept table,
    flags on the original rows, index-aligned to them).
    """
    originals = table[table["augment"] == ORIGINAL]
    _, flags = remove_outliers(originals, config)
    dropped_ids = set(originals.loc[flags, "id"])
    return table.loc[~table["id"].isin(dropped_ids)], flags
