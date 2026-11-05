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


# --- pass marks (thesis §4.10, ADR-013 item 4) ---


def per_fold_class_metrics(preds: pd.DataFrame, label: str) -> pd.DataFrame:
    """One row per (fold, class): precision and recall on that held-out fold.

    `preds` holds one model's out-of-fold predictions (columns fold, `label`,
    predicted). A class absent from a fold's validation set gets no row.
    """
    rows = []
    for fold, group in preds.groupby("fold"):
        per_class = compute_metrics(group[label], group["predicted"])["per_class"]
        for cls, m in per_class.items():
            if m["support"] > 0:
                rows.append({"fold": fold, "class": cls, "precision": m["precision"], "recall": m["recall"]})
    return pd.DataFrame(rows)


def check_pass_marks(
    predictions: pd.DataFrame,
    fold_scores: dict[str, list[float]],
    selected: str,
    marks: dict,
    label: str,
    baseline: str = "logistic_regression",
) -> dict:
    """Check the selected model against the team's fixed targets.

    Every target is reported with its actual value; a miss is reported, never
    tuned away (thesis §4.10). The baseline gap compares mean outer-fold
    Macro F1 of the selected ensemble against Logistic Regression and needs
    to exceed the selected model's outer-fold standard deviation.
    """
    scores = pd.Series(fold_scores[selected])
    per_fold = per_fold_class_metrics(predictions[predictions["model"] == selected], label)
    recall = per_fold.groupby("class")["recall"].mean()
    precision = per_fold.groupby("class")["precision"].mean()
    focus = marks["precision_class"]

    checks = {
        "macro_f1": {
            "value": round(float(scores.mean()), 4),
            "target": marks["macro_f1"],
            "stretch": marks.get("macro_f1_stretch"),
            "passed": bool(scores.mean() >= marks["macro_f1"]),
        },
        "min_class_recall": {
            "value": round(float(recall.min()), 4),
            "worst_class": str(recall.idxmin()),
            "per_class": {c: round(float(v), 4) for c, v in recall.items()},
            "target": marks["min_class_recall"],
            "passed": bool(recall.min() >= marks["min_class_recall"]),
        },
        f"class_{focus}_precision": {
            "value": round(float(precision.get(focus, float("nan"))), 4),
            "target": marks["class_precision"],
            "passed": bool(precision.get(focus, -1.0) >= marks["class_precision"]),
        },
    }
    if baseline in fold_scores:
        gap = scores.mean() - pd.Series(fold_scores[baseline]).mean()
        sd = scores.std(ddof=1) if len(scores) > 1 else 0.0
        checks["beats_baseline"] = {
            "value": round(float(gap), 4),
            "target": f"> {round(float(sd), 4)} (one outer-fold SD of {selected})",
            "passed": bool(gap > sd),
        }
    return {"selected": selected, "checks": checks,
            "all_passed": all(c["passed"] for c in checks.values())}
