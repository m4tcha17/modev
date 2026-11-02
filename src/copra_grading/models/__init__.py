"""Four candidate models + Optuna tuning. process.md Step 5.

All four are natively class-weighted, including the Logistic Regression
baseline (deliberate - keeps comparison isolating algorithm strength, not
which models got imbalance help). Optuna optimizes Macro F1, never raw
accuracy. Regularization tuned: max depth + min samples per leaf (RF),
max depth + min child weight (XGBoost, LightGBM). Search spaces/budgets:
see ADR-003.
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from copra_grading.features.table import feature_columns
from copra_grading.models import lightgbm_model, logreg, random_forest, xgboost_model

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
_BUILDERS = {
    "logistic_regression": logreg.build_model,
    "random_forest": random_forest.build_model,
    "xgboost": xgboost_model.build_model,
    "lightgbm": lightgbm_model.build_model,
}
LABEL = "copra_class"


def build_model(name: str, params: dict, n_jobs: int = 1, random_state: int = 42):
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return _BUILDERS[name](params, n_jobs=n_jobs, random_state=random_state)


def fit_predict(
    name: str,
    params: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_jobs: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """Fit one model on a fold's training rows (originals + augmented) and
    return class-letter predictions for its validation rows.

    Labels are encoded to 0..k-1 in sorted order (XGBoost needs contiguous
    integer labels) and decoded back. Logistic Regression gets features
    standardized on the training rows only.
    """
    cols = feature_columns(train)
    X_train, X_val = train[cols], val[cols]
    classes = np.array(sorted(train[LABEL].astype(str).unique()))
    y_train = np.searchsorted(classes, train[LABEL].astype(str).to_numpy())

    if name == "logistic_regression":
        X_train, X_val = logreg.scale_features(X_train, X_val)

    model = build_model(name, params, n_jobs=n_jobs, random_state=random_state)
    if name == "xgboost":
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    return classes[np.asarray(model.predict(X_val)).astype(int).ravel()]
