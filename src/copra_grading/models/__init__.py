"""Four candidate models + Optuna tuning. process.md Step 5.

All four are natively class-weighted, including the Logistic Regression
baseline (deliberate - keeps comparison isolating algorithm strength, not
which models got imbalance help). Optuna optimizes Macro F1, never raw
accuracy. Regularization tuned: max depth + min samples per leaf (RF),
max depth + min child weight (XGBoost, LightGBM). Search spaces/budgets:
see ADR-003.
"""

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from copra_grading.models import lightgbm_model, logreg, random_forest, xgboost_model

_MODULES = {
    "logistic_regression": logreg,
    "random_forest": random_forest,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}
MODEL_NAMES = tuple(_MODULES)


@dataclass(frozen=True)
class FittedModel:
    """A trained estimator plus the class letters behind its integer labels."""

    name: str
    estimator: object
    encoder: LabelEncoder

    @property
    def classes(self) -> np.ndarray:
        return self.encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.encoder.inverse_transform(self.estimator.predict(X).astype(int))


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    return _MODULES[name].suggest_params(trial)


def fit_model(name: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    """Fit one model on integer-encoded labels. XGBoost gets balanced
    per-row sample weights; the others carry class_weight='balanced'."""
    encoder = LabelEncoder().fit(y)
    y_int = encoder.transform(y)
    estimator = _MODULES[name].build_model(params, seed)
    if name == "xgboost":
        estimator.fit(X, y_int, sample_weight=compute_sample_weight("balanced", y_int))
    else:
        estimator.fit(X, y_int)
    return FittedModel(name, estimator, encoder)
