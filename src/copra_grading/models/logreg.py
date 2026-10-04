"""Logistic Regression - baseline only, never eligible for deployment. process.md Step 5.

Requires standardized feature scaling (only model in this pipeline that does -
see ../../../CLAUDE.md).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_model(params: dict) -> LogisticRegression:
    """Construct a class_weight='balanced' Logistic Regression from tuned params."""
    raise NotImplementedError


def scale_features(train: pd.DataFrame, *others: pd.DataFrame):
    """Standardize features, fit on train only, apply same transform to others."""
    raise NotImplementedError
