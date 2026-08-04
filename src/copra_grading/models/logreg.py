"""Logistic Regression - baseline only, never eligible for deployment. Spec §8a.

Requires standardized feature scaling (only model in this pipeline that does -
see ../../../CLAUDE.md / instructions §7c).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_model(params: dict) -> LogisticRegression:
    """Construct a class_weight='balanced' Logistic Regression from tuned params."""
    raise NotImplementedError


def scale_features(train: pd.DataFrame, *others: pd.DataFrame):
    """Standardize features, fit on train only, apply same transform to others."""
    raise NotImplementedError
