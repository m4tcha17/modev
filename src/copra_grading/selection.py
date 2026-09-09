"""Algorithm selection (M-BM-'8c step 1) vs. deployment configuration (M-BM-'9) - two
distinct stages, never conflated. See ../../../CLAUDE.md.
"""

import pandas as pd

# --- algorithm_selection ---


def select_algorithm(
    X_combined: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> str:
    """Train/tune all four models on the combined all-angle feature set,
    return the name of the best-scoring ensemble ("random_forest" | "xgboost" | "lightgbm").
    """
    raise NotImplementedError


# --- deployment_config ---


def evaluate_angle_configs(
    selected_algorithm: str,
    features_by_angle: dict[str, pd.DataFrame],
    y: pd.Series,
    groups: pd.Series,
) -> dict[str, float]:
    """Retrain selected_algorithm on each angle config, return Macro F1 per config."""
    raise NotImplementedError


def choose_deployment_config(angle_scores: dict[str, float], min_acceptable_f1: float) -> str:
    """Pick the smallest angle config meeting min_acceptable_f1; fall back toward
    larger combinations only if no single angle suffices - never default to all six.
    """
    raise NotImplementedError
