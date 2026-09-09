"""Duplicate Sample_ID resolution + feature-level outlier detection. Spec §6.

Completeness (all six images + moisture reading present) is guaranteed
upstream by Jotter - no imputation logic belongs here.

Outlier method/cutoff (IQR, 1.5x) is an ADR-001 default pending thesis-team
confirmation, not a fixed decision - see README.
"""

from dataclasses import dataclass, field

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


def flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    """Flag rows with any feature value outside Q1 - m*IQR .. Q3 + m*IQR."""
    raise NotImplementedError


def flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flag rows with any feature |z-score| above threshold."""
    raise NotImplementedError
