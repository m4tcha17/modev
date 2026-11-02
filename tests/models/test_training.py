import numpy as np
import pandas as pd
import pytest

from copra_grading.augmentation import TRANSFORMS
from copra_grading.evaluation import compute_metrics, fold_scores
from copra_grading.folds import make_folds
from copra_grading.models import MODEL_NAMES, fit_predict
from copra_grading.models.tuning import tune_model
from copra_grading.selection import cross_validate_models, select_model

CONFIG = {
    "splitting": {"n_splits": 3, "group_column": "batch_id",
                  "label_column": "copra_class", "random_state": 0},
    "augmentation": {"base_copies": 1, "max_multiplier": 4},
}


def _table(n_batches=18, seed=0):
    """Separable synthetic data: class D has a shifted feature mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batches):
        cls = ("A", "B", "D")[b % 3]
        for p in range(4):
            base = {"A": 0.0, "B": 3.0, "D": 6.0}[cls] + rng.normal(0, 0.3)
            for t in TRANSFORMS:
                rows.append({"id": f"B{b}-{p}", "batch_id": f"B{b}", "copra_class": cls,
                             "augment": t, "mask_fraction": 0.5,
                             "f1": base + rng.normal(0, 0.1), "f2": rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def folds():
    return make_folds(_table(), CONFIG)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_class_letters(name, folds):
    fold = folds[0]
    pred = fit_predict(name, {}, fold.train, fold.val)
    assert len(pred) == len(fold.val)
    assert set(pred) <= {"A", "B", "D"}
    # separable data: each model should get this right
    assert (pred == fold.val["copra_class"].to_numpy()).mean() > 0.9


def test_fit_predict_ignores_meta_columns(folds):
    """mask_fraction etc. must not be features: scrambling it changes nothing."""
    fold = folds[0]
    train = fold.train.assign(mask_fraction=np.arange(len(fold.train)))
    a = fit_predict("random_forest", {"n_estimators": 20}, fold.train, fold.val)
    b = fit_predict("random_forest", {"n_estimators": 20}, train, fold.val)
    assert (a == b).all()


def test_tune_model_records_fold_scores(folds):
    study = tune_model("logistic_regression", folds, n_trials=2)
    assert len(study.trials) == 2
    assert len(study.best_trial.user_attrs["fold_scores"]) == len(folds)
    assert 0.0 <= study.best_value <= 1.0


def test_cross_validate_predicts_every_val_photo_once(folds):
    oof = cross_validate_models(folds, {"lightgbm": {"n_estimators": 20}})["lightgbm"]
    assert not oof["id"].duplicated().any()
    assert set(oof["id"]) == set(pd.concat(f.val["id"] for f in folds))
    assert len(fold_scores(oof)) == len(folds)


def test_select_model_never_picks_logistic_regression():
    scores = {"logistic_regression": [0.99, 0.99], "random_forest": [0.7, 0.8],
              "xgboost": [0.8, 0.8], "lightgbm": [0.6, 0.6]}
    assert select_model(scores) == "xgboost"


def test_compute_metrics_moisture_order_and_absent_classes():
    y_true = pd.Series(["B", "A", "D", "D", "A"])
    y_pred = pd.Series(["B", "A", "D", "B", "A"])
    m = compute_metrics(y_true, y_pred)
    # A before B before D (moisture order F,A,B,C,D,E); C/E/F not present
    assert list(m["confusion_matrix"].index) == ["A", "B", "D"]
    assert m["confusion_matrix"].loc["D", "B"] == 1
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["per_class"].loc["D", "recall"] == pytest.approx(0.5)
    assert m["per_class"].loc["B", "precision"] == pytest.approx(0.5)
    expected_f1 = np.mean([1.0, 2 / 3, 2 / 3])
    assert m["macro_f1"] == pytest.approx(expected_f1)
