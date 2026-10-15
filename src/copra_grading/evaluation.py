"""Evaluation. process.md Step 7. Report every metric below, per photo, on the held-out folds - never a single accuracy number."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from copra_grading.dataset import MOISTURE_ORDER

# --- metrics ---


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, Macro F1, and per-class precision
    and recall. Order confusion-matrix rows/columns by dataset.MOISTURE_ORDER
    so errors between neighbouring moisture bands sit next to the diagonal.
    """
    present = set(y_true) | set(y_pred)
    labels = [c for c in MOISTURE_ORDER if c in present]
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p), "recall": float(r), "support": int(s)}
            for c, p, r, s in zip(labels, precision, recall, support)
        },
    }
