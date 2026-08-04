"""Optuna hyperparameter tuning. Spec §8b. Search spaces/budget: see ADR-003.

Objective is Macro F1 for every model, not raw accuracy - minority-class
(Class 1 precision, Class 3 recall) performance is what matters practically,
and plain accuracy would let majority-class Class 2 dominate the score.
"""

import optuna
import pandas as pd


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> optuna.Study:
    """Run an Optuna study for the named model, optimizing cross-validated Macro F1."""
    raise NotImplementedError
