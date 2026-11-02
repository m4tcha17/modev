"""Evaluation. process.md Step 7. Report every metric below, per photo, on the held-out folds - never a single accuracy number."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from copra_grading.dataset import MOISTURE_ORDER

# --- metrics ---


def _labels(y_true: pd.Series, y_pred: pd.Series) -> list[str]:
    """Classes present in either series, in moisture order. Classes absent
    from the data (e.g. not yet collected) are left out, so they neither
    count as zero-F1 classes nor pad the confusion matrix."""
    present = set(map(str, y_true)) | set(map(str, y_pred))
    return [c for c in MOISTURE_ORDER if c in present]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, Macro F1, and per-class precision
    and recall. Order confusion-matrix rows/columns by dataset.MOISTURE_ORDER
    so errors between neighbouring moisture bands sit next to the diagonal.
    """
    y_true = pd.Series(y_true).astype(str).to_numpy()
    y_pred = pd.Series(y_pred).astype(str).to_numpy()
    labels = _labels(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=pd.Index(labels, name="true"),
            columns=pd.Index(labels, name="pred"),
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support},
            index=pd.Index(labels, name="class"),
        ),
    }


def fold_scores(oof: pd.DataFrame) -> list[float]:
    """Macro F1 per fold from out-of-fold predictions (selection.py)."""
    return [
        compute_metrics(g["y_true"], g["y_pred"])["macro_f1"]
        for _, g in oof.groupby("fold", sort=True)
    ]
