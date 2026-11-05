"""Model selection. process.md Step 6.

Compare all four models on Macro F1 across the 5 StratifiedGroupKFold folds. The best
of Random Forest / XGBoost / LightGBM is the selected (deployed) model.
Logistic Regression is a reference point only, never selectable.

Nested CV: for each outer fold, every model is tuned on that fold's training
photos only (tuning.tune_model), refitted on the fold's augmented training
rows with the best params, and scored on the fold's validation originals.

After selection, fit_final_model re-tunes the selected model on the full
dataset (grouped 5-fold) and trains it on every photo plus its augmented
copies. That model is the deployed one (REQ-4.2-7, ADR-013 item 5); its
score is the nested-CV estimate above, never a score on its own training data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.augmentation import ORIGINAL, class_multipliers, select_training_rows
from copra_grading.features.table import feature_columns
from copra_grading.folds import make_folds
from copra_grading.models import FittedModel, fit_model
from copra_grading.models.tuning import tune_model

ENSEMBLE_CANDIDATES = ("random_forest", "xgboost", "lightgbm")


@dataclass(frozen=True)
class CVResult:
    scores: dict[str, list[float]]
    best_params: dict[str, list[dict]]
    predictions: pd.DataFrame  # id, batch_id, copra_class, fold, model, predicted


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
    features = feature_columns(table)
    scores = {name: [] for name in model_names}
    best_params = {name: [] for name in model_names}
    predictions = []

    for fold in make_folds(table, config):
        train_photos = table[table["id"].isin(fold.train["id"])]
        for name in model_names:
            study = tune_model(name, train_photos, config, n_trials, seed + fold.index)
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

    return CVResult(scores, best_params, pd.concat(predictions, ignore_index=True))


def select_model(fold_scores: dict[str, list[float]]) -> str:
    """Return the ensemble in ENSEMBLE_CANDIDATES with the highest mean Macro F1."""
    candidates = {n: np.mean(s) for n, s in fold_scores.items() if n in ENSEMBLE_CANDIDATES}
    if not candidates:
        raise ValueError("no ensemble model in fold_scores - Logistic Regression is never selectable")
    return max(candidates, key=candidates.get)


def full_training_rows(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Every original photo plus augmented copies, with multipliers from the
    whole dataset's class counts - the same rule a training fold uses (ADR-002)."""
    label = config["splitting"]["label_column"]
    aug = config["augmentation"]
    originals = table[table["augment"] == ORIGINAL]
    counts = {str(c): int(n) for c, n in originals[label].value_counts().items()}
    multipliers = class_multipliers(counts, aug["base_copies"], aug["max_multiplier"])
    return select_training_rows(table, originals["id"], multipliers,
                                seed=config["splitting"]["random_state"])


def fit_final_model(
    table: pd.DataFrame, config: dict, model_name: str, n_trials: int
) -> tuple[FittedModel, dict]:
    """Re-tune `model_name` on the full dataset with a grouped
    n_splits-fold split (splitting.n_splits, 5), then train it on all
    photos plus augmented copies. Return (model, best params)."""
    if model_name not in ENSEMBLE_CANDIDATES:
        raise ValueError(f"{model_name} is not deployable - only {ENSEMBLE_CANDIDATES}")
    label = config["splitting"]["label_column"]
    seed = config["tuning"]["random_state"]
    study = tune_model(model_name, table, config, n_trials, seed,
                       n_splits=config["splitting"]["n_splits"])
    train = full_training_rows(table, config)
    model = fit_model(model_name, study.best_params, train[feature_columns(table)],
                      train[label], seed)
    return model, study.best_params
