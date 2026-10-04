"""Evaluation. process.md Step 7. Report every metric below, per photo, on the held-out folds - never a single accuracy number."""

import pandas as pd

# --- metrics ---


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, Macro F1, and per-class precision
    and recall. Order confusion-matrix rows/columns by dataset.MOISTURE_ORDER
    so errors between neighbouring moisture bands sit next to the diagonal.
    """
    raise NotImplementedError
