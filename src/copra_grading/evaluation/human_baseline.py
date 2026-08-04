"""Benchmark against experienced buying-station personnel using the manual
pasa method. Spec §11.

Accepts a human-labeled comparison file/column as input - build to expect
this is provided, not assume it's absent.
"""

import pandas as pd


def compare_to_human_baseline(
    y_true: pd.Series, y_pred_model: pd.Series, y_pred_human: pd.Series
) -> dict:
    """Return model accuracy vs. human accuracy on the same validation subset."""
    raise NotImplementedError
