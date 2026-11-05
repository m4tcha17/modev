import pandas as pd
import pytest

from copra_grading.evaluation import compute_metrics


def test_labels_follow_moisture_order_not_letters():
    m = compute_metrics(pd.Series(["E", "A", "F", "B"]), pd.Series(["E", "A", "F", "B"]))
    assert m["labels"] == ["F", "A", "B", "E"]
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_confusion_matrix_rows_true_columns_predicted():
    m = compute_metrics(pd.Series(["A", "A", "B"]), pd.Series(["A", "B", "B"]))
    assert m["confusion_matrix"] == [[1, 1], [0, 1]]
    assert m["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert m["per_class"]["B"]["precision"] == pytest.approx(0.5)


def test_class_never_predicted_has_zero_recall():
    m = compute_metrics(pd.Series(["A", "C", "C"]), pd.Series(["A", "A", "A"]))
    assert m["labels"] == ["A", "C"]
    assert m["per_class"]["C"] == {"precision": 0.0, "recall": 0.0, "support": 2}


MARKS = {"macro_f1": 0.75, "macro_f1_stretch": 0.80, "min_class_recall": 0.60,
         "precision_class": "A", "class_precision": 0.85}


def _preds(model, pairs, folds=2):
    """pairs: (true, predicted) repeated in every fold."""
    import pandas as pd

    rows = [{"fold": k, "model": model, "copra_class": t, "predicted": p}
            for k in range(folds) for t, p in pairs]
    return pd.DataFrame(rows)


def test_pass_marks_all_pass():
    from copra_grading.evaluation import check_pass_marks

    preds = _preds("lightgbm", [("A", "A"), ("B", "B"), ("C", "C")] * 4)
    result = check_pass_marks(preds, {"lightgbm": [0.9, 0.92], "logistic_regression": [0.7, 0.72]},
                              "lightgbm", MARKS, "copra_class")
    assert result["all_passed"]
    assert set(result["checks"]) == {"macro_f1", "min_class_recall", "class_A_precision", "beats_baseline"}


def test_pass_marks_report_misses_with_values():
    from copra_grading.evaluation import check_pass_marks

    # B is mostly called A: B recall 0.25, A precision 4/7.
    pairs = [("A", "A")] * 4 + [("B", "A")] * 3 + [("B", "B")] + [("C", "C")] * 4
    result = check_pass_marks(_preds("lightgbm", pairs), {"lightgbm": [0.70, 0.74], "logistic_regression": [0.78, 0.77]},
                              "lightgbm", MARKS, "copra_class")
    checks = result["checks"]
    assert not result["all_passed"]
    assert not checks["macro_f1"]["passed"] and checks["macro_f1"]["value"] == 0.72
    assert checks["min_class_recall"]["worst_class"] == "B" and checks["min_class_recall"]["value"] == 0.25
    assert checks["class_A_precision"]["value"] == round(4 / 7, 4) and not checks["class_A_precision"]["passed"]
    assert checks["beats_baseline"]["value"] < 0 and not checks["beats_baseline"]["passed"]


def test_baseline_gap_must_exceed_one_sd():
    from copra_grading.evaluation import check_pass_marks

    preds = _preds("xgboost", [("A", "A"), ("B", "B")] * 4)
    # gap 0.02, SD of [0.80, 0.90] = 0.0707 -> not enough
    result = check_pass_marks(preds, {"xgboost": [0.80, 0.90], "logistic_regression": [0.83, 0.83]},
                              "xgboost", MARKS, "copra_class")
    assert not result["checks"]["beats_baseline"]["passed"]
