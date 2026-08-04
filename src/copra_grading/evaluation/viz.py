"""PCA / t-SNE sanity-check visualization of the combined feature space. Spec §11.

If the three classes overlap heavily here, that's a feature-engineering
problem, not something more model tuning will fix - flag it, don't keep tuning.
"""

import pandas as pd


def plot_feature_space(X_combined: pd.DataFrame, y: pd.Series, method: str = "pca"):
    """Project combined feature space to 2D (pca | tsne) and plot colored by class."""
    raise NotImplementedError
