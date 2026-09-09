"""Duplicate Sample_ID resolution + feature-level outlier detection. Spec §6.

Completeness (all six images + moisture reading present) is guaranteed
upstream by Jotter - no imputation logic belongs here.

Outlier method/cutoff (IQR, 1.5x) is an ADR-001 default pending thesis-team
confirmation, not a fixed decision - see README.
"""

import pandas as pd

# --- §6.1 Duplicate Sample_ID resolution --------------------------------------


def resolve_duplicate_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and resolve duplicate Sample_ID rows across the merged dataset."""
    raise NotImplementedError


# --- §6.2 Feature-level outlier detection (ADR-001) --------------------------


def flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    """Flag rows with any feature value outside Q1 - m*IQR .. Q3 + m*IQR."""
    raise NotImplementedError


def flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flag rows with any feature |z-score| above threshold."""
    raise NotImplementedError
