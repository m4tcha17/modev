"""Near-boundary vs. mid-range bucketed evaluation. Spec §11. Tolerance band: see ADR-007.

Ground-truth measurement variance (Brown-Duvel meter) matters most exactly at
the 6.0/6.1 and 13.9/14.0 boundaries - report these buckets separately, never
folded into one overall accuracy number.
"""

import pandas as pd

from copra_grading.labels import LOWER_THRESHOLD, UPPER_THRESHOLD


def bucket_by_boundary_proximity(
    moisture_readings: pd.Series, tolerance_pct: float = 0.5
) -> pd.Series:
    """Return "near_boundary" | "mid_range" per sample, based on distance to
    LOWER_THRESHOLD or UPPER_THRESHOLD.
    """
    raise NotImplementedError
