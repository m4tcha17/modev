"""Optuna hyperparameter tuning. process.md Step 5. Search spaces/budget: see ADR-003.

Objective is Macro F1 for every model, not raw accuracy - plain accuracy
would let the majority classes among A-F dominate the score.

Nested CV: tune_model only ever receives the photos of one outer training
fold. It re-splits them with folds.make_folds (grouped, stratified, split
before augment) into tuning.inner_n_splits inner folds and scores each trial
on the inner validation originals.
"""

import copy

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.features.table import feature_columns
from copra_grading.folds import make_folds
from copra_grading.models import fit_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_config(config: dict) -> dict:
    """Copy of config with the inner split count."""
    inner = copy.deepcopy(config)
    inner["splitting"]["n_splits"] = config["tuning"]["inner_n_splits"]
    return inner


def tune_model(
    model_name: str,
    table: pd.DataFrame,
    config: dict,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run an Optuna study for the named model, maximising mean Macro F1
    over grouped + stratified inner folds of `table` (never ungrouped CV -
    leaks parts of one whole sample)."""
    label = config["splitting"]["label_column"]
    features = feature_columns(table)
    inner_folds = make_folds(table, inner_config(config))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        scores = []
        for fold in inner_folds:
            model = fit_model(model_name, params, fold.train[features], fold.train[label], seed)
            pred = model.predict(fold.val[features])
            scores.append(f1_score(fold.val[label], pred, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study
