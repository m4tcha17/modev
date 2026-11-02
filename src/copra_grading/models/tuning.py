"""Optuna hyperparameter tuning. process.md Step 5. Search spaces/budget: see ADR-003.

Objective is Macro F1 for every model, not raw accuracy - plain accuracy
would let the majority classes among A-F dominate the score.
"""

import optuna
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.models import LABEL, fit_predict


def suggest_params(model_name: str, trial: optuna.Trial) -> dict:
    """ADR-003 search spaces. LightGBM also gets an explicit max_depth and
    min_child_weight, as ADR-003 requires (regularization tuning)."""
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"unknown model {model_name!r}")


def fold_macro_f1(model_name: str, params: dict, folds: list, n_jobs: int = 1) -> list[float]:
    """Macro F1 on each fold's validation photos, model trained on that
    fold's training rows (originals + augmented copies)."""
    scores = []
    for fold in folds:
        pred = fit_predict(model_name, params, fold.train, fold.val, n_jobs=n_jobs)
        scores.append(float(f1_score(fold.val[LABEL].astype(str), pred, average="macro")))
    return scores


def tune_model(
    model_name: str,
    folds: list,
    n_trials: int,
    n_jobs: int = 1,
    seed: int = 42,
) -> optuna.Study:
    """Run an Optuna study for the named model, maximizing mean Macro F1
    over `folds` - the StratifiedGroupKFold folds from folds.make_folds,
    grouped by whole sample, augmented copies in training folds only (never
    ungrouped CV - leaks parts of one whole sample). Per-fold scores are
    kept on each trial as the `fold_scores` user attribute."""

    def objective(trial: optuna.Trial) -> float:
        scores = fold_macro_f1(model_name, suggest_params(model_name, trial), folds, n_jobs)
        trial.set_user_attr("fold_scores", scores)
        return float(pd.Series(scores).mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=model_name,
    )
    study.optimize(objective, n_trials=n_trials)
    return study
