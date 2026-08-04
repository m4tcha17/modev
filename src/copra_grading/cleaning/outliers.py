"""Feature-level outlier detection. Spec §6.2. Method/cutoff: see ADR-001.

Runs on extracted feature values (post feature-extraction), not raw images -
flags rows that passed file-level completeness but would still produce
unreliable features (glare, unexpected object in frame, etc.).
"""

import pandas as pd


def flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    """Flag rows with any feature value outside Q1 - m*IQR .. Q3 + m*IQR."""
    raise NotImplementedError


def flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flag rows with any feature |z-score| above threshold."""
    raise NotImplementedError
