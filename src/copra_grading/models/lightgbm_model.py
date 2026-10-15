"""LightGBM ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

import optuna
from lightgbm import LGBMClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space, plus max_depth and min_child_weight (process.md)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_model(params: dict, seed: int) -> LGBMClassifier:
    """Construct a class_weight='balanced' LightGBM classifier from tuned params.
    deterministic + force_row_wise: same results on any core count."""
    return LGBMClassifier(
        class_weight="balanced", subsample_freq=1, random_state=seed,
        deterministic=True, force_row_wise=True,
        n_jobs=-1, verbose=-1, **params,
    )
