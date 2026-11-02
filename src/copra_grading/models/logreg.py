"""Logistic Regression - baseline only, never eligible for deployment. process.md Step 5.

Requires standardized feature scaling (only model in this pipeline that does -
see ../../../CLAUDE.md).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_model(params: dict, n_jobs: int = 1, random_state: int = 42) -> LogisticRegression:
    """Construct a class_weight='balanced' Logistic Regression from tuned params.

    saga supports both l1 and l2 penalties for multiclass. n_jobs is accepted
    for a uniform build_model signature; saga runs single-threaded.
    The ADR-003 `penalty` choice ("l1"/"l2") maps to `l1_ratio` 1/0, since
    scikit-learn >= 1.8 deprecates `penalty`.
    """
    params = dict(params)
    penalty = params.pop("penalty", "l2")
    return LogisticRegression(
        **params,
        l1_ratio={"l1": 1.0, "l2": 0.0}[penalty],
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )


def scale_features(train: pd.DataFrame, *others: pd.DataFrame):
    """Standardize features, fit on train only, apply same transform to others."""
    scaler = StandardScaler().fit(train)

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    return (_apply(train), *(_apply(df) for df in others))
