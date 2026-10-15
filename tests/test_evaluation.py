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
