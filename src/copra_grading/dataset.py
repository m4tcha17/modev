"""Dataset loading + schema checks. process.md Step 1.

Layout: `copra-dataset.csv` and `photos/` sit in the same directory. One CSV
row = one photo. Columns: `id`, `batch_id`, `copra_class`, `path`. The four
photos of one batch share a `batch_id`; several batches can come from the
same whole copra sample, which the CSV does not yet record (see
splitting.py). The label is read straight from `copra_class` - one
of A-F - never derived.

Batch checks are report-only: a BatchIssue is returned per problem, rows are
never dropped or edited here.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VALID_CLASSES = ("A", "B", "C", "D", "E", "F")

# Moisture-content band (%) behind each class - see docs/classes.md. For
# display/docs only: labels still come from the copra_class column, and
# boundary inclusivity (is 9.0 A or B?) is unconfirmed.
CLASS_DESCRIPTIONS = {
    "F": "below 6% moisture - penalty (over-dried)",
    "A": "6-9% moisture",
    "B": "9-12% moisture",
    "C": "12-15% moisture",
    "D": "15-18% moisture",
    "E": "18% moisture and above - lowest payout/quality",
}

# Classes sorted by increasing moisture. Letters are NOT in moisture order:
# F is the driest, E the wettest.
MOISTURE_ORDER = ("F", "A", "B", "C", "D", "E")
REQUIRED_COLUMNS = ("id", "batch_id", "copra_class", "path")
PHOTOS_PER_BATCH = 4


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the CSV, check required columns and class values.

    `copra_class` is coerced to str. Raises ValueError on missing columns or
    a class outside VALID_CLASSES.
    """
    df = pd.read_csv(csv_path, dtype={"id": str, "batch_id": str, "copra_class": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    bad = sorted(set(df["copra_class"]) - set(VALID_CLASSES))
    if bad:
        raise ValueError(f"unknown copra_class values: {bad} (expected one of {VALID_CLASSES})")
    return df


def load_image(csv_path: Path, relative_path: str) -> np.ndarray:
    """Load one photo as an (H, W, 3) RGB uint8 array.

    `relative_path` is the CSV's `path` column, resolved against the CSV's
    own directory.
    """
    full_path = Path(csv_path).parent / relative_path
    bgr = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {full_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --- batch checks ------------------------------------------------------------


@dataclass
class BatchIssue:
    batch_id: str
    reason: str  # "row_count" | "class_conflict" | "duplicate_id"
    detail: str
    row_labels: list = field(default_factory=list)


@dataclass
class DatasetReport:
    issues: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


class DatasetError(Exception):
    def __init__(self, report: "DatasetReport") -> None:
        self.report = report
        lines = [f"  {i.batch_id}: {i.reason} - {i.detail}" for i in report.issues]
        super().__init__("dataset batch issues:\n" + "\n".join(lines))


def check_batches(df: pd.DataFrame) -> DatasetReport:
    """Report batches with a row count != PHOTOS_PER_BATCH, more than one
    distinct copra_class, or a repeated photo `id`.
    """
    issues: list = []
    for batch_id, group in df.groupby("batch_id", sort=True):
        batch_id = str(batch_id)
        labels = list(group.index)

        if len(group) != PHOTOS_PER_BATCH:
            issues.append(
                BatchIssue(
                    batch_id,
                    "row_count",
                    f"{len(group)} rows, expected {PHOTOS_PER_BATCH}",
                    labels,
                )
            )

        classes = sorted(str(c) for c in group["copra_class"].dropna().unique())
        if len(classes) > 1:
            issues.append(
                BatchIssue(
                    batch_id,
                    "class_conflict",
                    f"{len(classes)} distinct copra_class values: {classes}",
                    labels,
                )
            )

        dup_mask = group["id"].duplicated(keep=False)
        if dup_mask.any():
            repeated = sorted({str(i) for i in group.loc[dup_mask, "id"]})
            issues.append(
                BatchIssue(
                    batch_id,
                    "duplicate_id",
                    f"repeated id: {', '.join(repeated)}",
                    list(group.index[dup_mask]),
                )
            )

    return DatasetReport(issues)


def assert_clean(df: pd.DataFrame) -> None:
    """Raise DatasetError if check_batches finds any issue."""
    report = check_batches(df)
    if not report.is_clean:
        raise DatasetError(report)
