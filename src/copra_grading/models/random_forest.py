"""Random Forest ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

import optuna
from sklearn.ensemble import RandomForestClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def build_model(params: dict, seed: int) -> RandomForestClassifier:
    """Construct a class_weight='balanced' Random Forest from tuned params."""
    return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
