"""Evaluation suite. Spec M-BM-'11. Report every metric below - never settle for a single accuracy number."""

import pandas as pd

from copra_grading.labels import LOWER_THRESHOLD, UPPER_THRESHOLD

# --- metrics ---


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, F1, per-class precision/recall, Macro F1."""
    raise NotImplementedError


# --- boundary_analysis ---


def bucket_by_boundary_proximity(
    moisture_readings: pd.Series, tolerance_pct: float = 0.5
) -> pd.Series:
    """Return "near_boundary" | "mid_range" per sample, based on distance to
    LOWER_THRESHOLD or UPPER_THRESHOLD.
    """
    raise NotImplementedError


# --- human_baseline ---


def compare_to_human_baseline(
    y_true: pd.Series, y_pred_model: pd.Series, y_pred_human: pd.Series
) -> dict:
    """Return model accuracy vs. human accuracy on the same validation subset."""
    raise NotImplementedError


# --- ablation ---


def run_ablation(
    X_by_family: dict[str, pd.DataFrame], y: pd.Series, groups: pd.Series
) -> dict[str, float]:
    """Train the selected algorithm on each family subset and combined; return Macro F1 per config."""
    raise NotImplementedError


# --- viz ---


def plot_feature_space(X_combined: pd.DataFrame, y: pd.Series, method: str = "pca"):
    """Project combined feature space to 2D (pca | tsne) and plot colored by class."""
    raise NotImplementedError
