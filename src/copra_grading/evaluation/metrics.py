"""Core classification metrics. Spec §11.

Class 1 precision and Class 3 recall get particular emphasis in reporting - a
Class 1 false positive is a direct financial loss (wet copra sold as premium);
a Class 3 false negative is a health/liability risk (aflatoxin/mold missed).
"""

import pandas as pd


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, F1, per-class precision/recall, Macro F1."""
    raise NotImplementedError
