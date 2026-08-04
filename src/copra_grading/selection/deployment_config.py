"""Deployment configuration: angle-wise retrain of the selected algorithm only. Spec §9.

Retrains ONLY the algorithm selection.py picked - never all four again - on
each of: top-only, each side individually, all-sides-combined, bottom-only,
and combined-all-angle (kept purely as the upper-bound comparison point).
Whichever single-angle (or smallest angle-subset) configuration reaches
acceptable accuracy becomes the deployed model's required input - a live user
should submit as few photos as possible, ideally one.

This result is a hard input to artifact/serialize.py, not just a reported metric.
"""

import pandas as pd


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
