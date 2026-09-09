"""Duplicate Sample_ID resolution + feature-level outlier detection. Spec §6.

Completeness (all six images + moisture reading present) is guaranteed
upstream by Jotter - no imputation logic belongs here.

Outlier method/cutoff (IQR, 1.5x multiplier, min_flagged_features=3) is an
ADR-001 default pending thesis-team confirmation, not a fixed decision - see
README. Outlier flagging is advisory: it returns a boolean Series aligned to
the input index and drops nothing.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- §6.1 Duplicate Sample_ID resolution --------------------------------------

ANGLES_PER_SAMPLE = 6


@dataclass
class SampleConflict:
    sample_id: str
    reason: str  # "row_count" | "duplicate_angle" | "moisture_conflict"
    detail: str
    row_labels: list = field(default_factory=list)


@dataclass
class DedupReport:
    conflicts: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.conflicts) == 0


class DuplicateSampleError(Exception):
    def __init__(self, report: "DedupReport") -> None:
        self.report = report
        lines = [
            f"  {c.sample_id}: {c.reason} - {c.detail}" for c in report.conflicts
        ]
        super().__init__("duplicate Sample_ID conflicts:\n" + "\n".join(lines))


def resolve_duplicate_samples(df: pd.DataFrame) -> DedupReport:
    """Detect (never resolve destructively) conflicting Sample_ID groups.

    A conflict is reported, one SampleConflict per issue, when a Sample_ID
    group has: a row count != ANGLES_PER_SAMPLE, a repeated Angle_ID, or more
    than one distinct moisture_reading (the offline-clerk ID-collision signal).
    Rows are never dropped or edited here.
    """
    conflicts: list = []
    for sample_id, group in df.groupby("Sample_ID", sort=True):
        sample_id = str(sample_id)
        labels = list(group.index)

        if len(group) != ANGLES_PER_SAMPLE:
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "row_count",
                    f"{len(group)} rows, expected {ANGLES_PER_SAMPLE}",
                    labels,
                )
            )

        dup_mask = group["Angle_ID"].duplicated(keep=False)
        if dup_mask.any():
            repeated = sorted({str(a) for a in group.loc[dup_mask, "Angle_ID"]})
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "duplicate_angle",
                    f"repeated Angle_ID: {', '.join(repeated)}",
                    list(group.index[dup_mask]),
                )
            )

        distinct_moisture = sorted(group["moisture_reading"].dropna().unique())
        if len(distinct_moisture) > 1:
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "moisture_conflict",
                    f"{len(distinct_moisture)} distinct moisture_reading values: {distinct_moisture}",
                    labels,
                )
            )

    return DedupReport(conflicts)


def assert_clean(df: pd.DataFrame) -> None:
    """Raise DuplicateSampleError if resolve_duplicate_samples finds any conflict."""
    report = resolve_duplicate_samples(df)
    if not report.is_clean:
        raise DuplicateSampleError(report)


# --- §6.2 Feature-level outlier detection (ADR-001) --------------------------


_IDENTIFIER_COLUMNS = ("Sample_ID", "Angle_ID", "moisture_reading")


def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    kept = [c for c in features.columns if c not in _IDENTIFIER_COLUMNS]
    return features[kept].select_dtypes(include="number")


def _row_flags(out_of_bounds: pd.DataFrame, min_flagged_features: int, index) -> pd.Series:
    counts = out_of_bounds.sum(axis=1)
    return (counts >= min_flagged_features).reindex(index, fill_value=False).astype(bool)


def flag_outliers_iqr(
    features: pd.DataFrame, multiplier: float = 1.5, min_flagged_features: int = 3
) -> pd.Series:
    """Flag a row when >= min_flagged_features of its numeric features fall
    outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]. Advisory only - nothing
    is dropped. Method/cutoff is an ADR-001 default pending confirmation.
    """
    numeric = _numeric_features(features)
    if numeric.empty:
        return pd.Series(False, index=features.index, dtype=bool)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    out_of_bounds = numeric.lt(lower, axis=1) | numeric.gt(upper, axis=1)
    return _row_flags(out_of_bounds, min_flagged_features, features.index)


def flag_outliers_zscore(
    features: pd.DataFrame, threshold: float = 3.0, min_flagged_features: int = 3
) -> pd.Series:
    """Flag a row when >= min_flagged_features of its numeric features have
    |z-score| > threshold. Zero-variance columns contribute nothing.
    """
    numeric = _numeric_features(features)
    if numeric.empty:
        return pd.Series(False, index=features.index, dtype=bool)
    std = numeric.std(ddof=0).replace(0.0, np.nan)
    z = (numeric - numeric.mean()).abs().div(std, axis=1)
    out_of_bounds = z.gt(threshold).fillna(False)
    return _row_flags(out_of_bounds, min_flagged_features, features.index)


def flag_outliers(features: pd.DataFrame, config: dict) -> pd.Series:
    """Dispatch to the method named in config["cleaning"]["outlier_method"]."""
    cleaning_cfg = config["cleaning"]
    method = cleaning_cfg["outlier_method"]
    min_flagged = cleaning_cfg.get("outlier_min_features", 3)
    if method == "iqr":
        return flag_outliers_iqr(
            features, cleaning_cfg["iqr_multiplier"], min_flagged
        )
    if method == "zscore":
        return flag_outliers_zscore(
            features, cleaning_cfg["zscore_threshold"], min_flagged
        )
    raise ValueError(f"unknown outlier_method: {method!r} (expected 'iqr' or 'zscore')")
