"""Redundant-feature pruning, fitted per training fold (ADR-012).

Many GLCM / color features are near-copies of each other (e.g. one GLCM
property at three distances). With ~235 samples, 65 features let the tree
ensembles fit noise. Greedy pass in extraction order: keep a feature unless
its |Spearman rho| with an already-kept feature exceeds the threshold.

Unsupervised (labels never used) and fitted on training rows only, so the
validation fold never influences which features are kept. Correlations use
original photos only - augmented copies are near-duplicates and would
overweight photos with more copies.
"""

import pandas as pd

from copra_grading.augmentation import ORIGINAL
from copra_grading.features.table import feature_columns


def select_uncorrelated(table: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    """Features of `table` to keep, in the order given."""
    originals = table[table["augment"] == ORIGINAL]
    # constant columns give NaN; treat as uncorrelated so they never block others
    corr = originals[features].corr(method="spearman").abs().fillna(0.0)
    kept: list[str] = []
    for name in features:
        if all(corr.loc[name, k] <= threshold for k in kept):
            kept.append(name)
    return kept


def selected_features(table: pd.DataFrame, config: dict) -> list[str]:
    """Apply `feature_selection.correlation_threshold` from the config to the
    training rows in `table`. Missing section or null threshold keeps all."""
    features = feature_columns(table)
    threshold = config.get("feature_selection", {}).get("correlation_threshold")
    if threshold is None:
        return features
    return select_uncorrelated(table, features, threshold)
