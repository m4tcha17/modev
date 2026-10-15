"""Model selection. process.md Step 6.

Compare all four models on Macro F1 across the 5 StratifiedGroupKFold folds. The best
of Random Forest / XGBoost / LightGBM is the selected (deployed) model.
Logistic Regression is a reference point only, never selectable.

Nested CV: for each outer fold, every model is tuned on that fold's training
photos only (tuning.tune_model), refitted on the fold's augmented training
rows with the best params, and scored on the fold's validation originals.
Redundant features are pruned per outer fold, on its training photos only,
and all four models use the same pruned list (ADR-012).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.feature_selection import selected_features
from copra_grading.folds import make_folds
from copra_grading.models import fit_model
from copra_grading.models.tuning import tune_model

ENSEMBLE_CANDIDATES = ("random_forest", "xgboost", "lightgbm")


@dataclass(frozen=True)
class CVResult:
    scores: dict[str, list[float]]
    best_params: dict[str, list[dict]]
    predictions: pd.DataFrame  # id, batch_id, copra_class, fold, model, predicted
    features: list[list[str]]  # pruned feature list per outer fold (ADR-012)


# --- selection ---


def cross_validate_models(
    table: pd.DataFrame, config: dict, model_names: tuple[str, ...], n_trials: int
) -> CVResult:
    """Nested CV over the outer folds of `table` (a cleaned feature table
    with every transform row). Return per-fold Macro F1, per-fold best
    params and out-of-fold predictions per model.
    """
    label = config["splitting"]["label_column"]
    group = config["splitting"]["group_column"]
    seed = config["tuning"]["random_state"]
    scores = {name: [] for name in model_names}
    best_params = {name: [] for name in model_names}
    predictions = []
    fold_features = []

    for fold in make_folds(table, config):
        train_photos = table[table["id"].isin(fold.train["id"])]
        features = selected_features(train_photos, config)
        fold_features.append(features)
        print(f"fold {fold.index}: {len(features)} features after pruning", flush=True)
        for name in model_names:
            study = tune_model(name, train_photos, config, n_trials, seed + fold.index,
                               features=features)
            model = fit_model(name, study.best_params, fold.train[features],
                              fold.train[label], seed + fold.index)
            pred = model.predict(fold.val[features])
            scores[name].append(
                float(f1_score(fold.val[label], pred, average="macro", zero_division=0))
            )
            best_params[name].append(study.best_params)
            predictions.append(
                fold.val[["id", group, label]].assign(fold=fold.index, model=name, predicted=pred)
            )
            print(f"fold {fold.index} {name}: macro F1 {scores[name][-1]:.3f}", flush=True)

    return CVResult(scores, best_params, pd.concat(predictions, ignore_index=True),
                    fold_features)


def select_model(fold_scores: dict[str, list[float]]) -> str:
    """Return the ensemble in ENSEMBLE_CANDIDATES with the highest mean Macro F1."""
    candidates = {n: np.mean(s) for n, s in fold_scores.items() if n in ENSEMBLE_CANDIDATES}
    if not candidates:
        raise ValueError("no ensemble model in fold_scores - Logistic Regression is never selectable")
    return max(candidates, key=candidates.get)
