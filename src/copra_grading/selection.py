"""Model selection. process.md Step 6.

Compare all four models on Macro F1 across the 5 StratifiedGroupKFold folds. The best
of Random Forest / XGBoost / LightGBM is the selected (deployed) model.
Logistic Regression is a reference point only, never selectable.
"""

import pandas as pd

ENSEMBLE_CANDIDATES = ("random_forest", "xgboost", "lightgbm")

# --- selection ---


def cross_validate_models(
    X: pd.DataFrame, y: pd.Series, folds: list, tuned_params: dict[str, dict]
) -> dict[str, list[float]]:
    """Train each model (LR + 3 ensembles) on every fold's training split
    (augmented), score on the held-out split. Return per-fold Macro F1 per model.
    """
    raise NotImplementedError


def select_model(fold_scores: dict[str, list[float]]) -> str:
    """Return the ensemble in ENSEMBLE_CANDIDATES with the highest mean Macro F1."""
    raise NotImplementedError
