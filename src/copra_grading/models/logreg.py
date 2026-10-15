"""Logistic Regression - baseline only, never eligible for deployment. process.md Step 5.

Requires standardized feature scaling (only model in this pipeline that does -
see ../../../CLAUDE.md). The scaler sits inside the pipeline, so it is fitted
on each training split only.
"""

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space. `penalty` l1/l2 is l1_ratio 1.0/0.0 (sklearn 1.9)."""
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
    }


def build_model(params: dict, seed: int) -> Pipeline:
    """StandardScaler + class_weight='balanced' Logistic Regression."""
    model = LogisticRegression(
        solver="saga", max_iter=5000, class_weight="balanced", random_state=seed, **params
    )
    return Pipeline([("scale", StandardScaler()), ("model", model)])
