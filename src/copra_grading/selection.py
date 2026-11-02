"""Model selection. process.md Step 6.

Compare all four models on Macro F1 across the 5 StratifiedGroupKFold folds. The best
of Random Forest / XGBoost / LightGBM is the selected (deployed) model.
Logistic Regression is a reference point only, never selectable.
"""

import pandas as pd

from copra_grading.models import LABEL, fit_predict

ENSEMBLE_CANDIDATES = ("random_forest", "xgboost", "lightgbm")

# --- selection ---


def out_of_fold_predictions(
    model_name: str, params: dict, folds: list, n_jobs: int = 1
) -> pd.DataFrame:
    """Train on every fold's training split (augmented), predict its held-out
    split. One row per validation photo: id, batch_id, fold, y_true, y_pred."""
    parts = []
    for fold in folds:
        pred = fit_predict(model_name, params, fold.train, fold.val, n_jobs=n_jobs)
        parts.append(pd.DataFrame({
            "id": fold.val["id"].to_numpy(),
            "batch_id": fold.val["batch_id"].to_numpy(),
            "fold": fold.index,
            "y_true": fold.val[LABEL].astype(str).to_numpy(),
            "y_pred": pred,
        }))
    return pd.concat(parts, ignore_index=True)


def cross_validate_models(
    folds: list, tuned_params: dict[str, dict], n_jobs: int = 1
) -> dict[str, pd.DataFrame]:
    """Out-of-fold predictions for each model (LR + 3 ensembles) with its
    tuned params. Per-fold Macro F1 comes from evaluation.fold_scores."""
    return {
        name: out_of_fold_predictions(name, params, folds, n_jobs)
        for name, params in tuned_params.items()
    }


def select_model(fold_scores: dict[str, list[float]]) -> str:
    """Return the ensemble in ENSEMBLE_CANDIDATES with the highest mean Macro F1."""
    candidates = {n: s for n, s in fold_scores.items() if n in ENSEMBLE_CANDIDATES}
    if not candidates:
        raise ValueError(f"no scores for any of {ENSEMBLE_CANDIDATES}")
    return max(candidates, key=lambda n: sum(candidates[n]) / len(candidates[n]))
