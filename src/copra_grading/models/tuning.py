"""Optuna hyperparameter tuning. process.md Step 5. Search spaces/budget: see ADR-003.

Objective is Macro F1 for every model, not raw accuracy - plain accuracy
would let the majority classes among A-F dominate the score.
"""

import optuna
import pandas as pd


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_trials: int,
) -> optuna.Study:
    """Run an Optuna study for the named model, optimizing Macro F1 over
    the same grouped + stratified folds as splitting.py (never ungrouped CV -
    leaks parts of one whole sample)."""
    raise NotImplementedError
