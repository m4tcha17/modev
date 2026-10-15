"""XGBoost ensemble candidate. process.md Step 5. No feature scaling needed (tree-based).

XGBoost has no class_weight argument: models.fit_model passes balanced
per-row sample weights instead.
"""

import optuna
from xgboost import XGBClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_model(params: dict, seed: int) -> XGBClassifier:
    """Construct an XGBoost classifier; labels must be integer-encoded."""
    return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1, **params)
