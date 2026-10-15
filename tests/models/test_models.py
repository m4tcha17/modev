import numpy as np
import optuna
import pandas as pd
import pytest

from copra_grading.models import MODEL_NAMES, fit_model, suggest_params


def _data(classes=("A", "B", "C", "D", "E"), n=40, seed=0, sizes=None):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, c in enumerate(classes):
        k = sizes[c] if sizes else n
        X.append(rng.normal(i * 4.0, 0.5, size=(k, 3)))
        y += [c] * k
    return pd.DataFrame(np.vstack(X), columns=["f1", "f2", "f3"]), pd.Series(y)


def _default_params(name):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return suggest_params(name, study.ask())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_class_letters_for_every_model(name):
    X, y = _data()
    model = fit_model(name, _default_params(name), X, y, seed=0)
    pred = model.predict(X)
    assert set(pred) <= {"A", "B", "C", "D", "E"}
    assert (pred == y.to_numpy()).mean() > 0.9


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_with_five_classes_predicts_string_labels(name):
    # class F missing entirely: model still trains on what is there
    X, y = _data(classes=("A", "C", "E"))
    model = fit_model(name, _default_params(name), X, y, seed=0)
    assert list(model.classes) == ["A", "C", "E"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_regularization_params_are_tuned(name):
    params = _default_params(name)
    required = {
        "random_forest": {"max_depth", "min_samples_leaf"},
        "xgboost": {"max_depth", "min_child_weight"},
        "lightgbm": {"max_depth", "min_child_weight"},
        "logistic_regression": {"C", "l1_ratio"},
    }[name]
    assert required <= set(params)


def test_class_weighting_lifts_minority_recall():
    # 200 A vs 10 B, overlapping: unweighted XGBoost would mostly ignore B
    rng = np.random.default_rng(1)
    X = pd.DataFrame(np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(1.0, 1, (10, 2))]),
                     columns=["f1", "f2"])
    y = pd.Series(["A"] * 200 + ["B"] * 10)
    model = fit_model("xgboost", {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
                                  "min_child_weight": 1.0, "subsample": 1.0,
                                  "colsample_bytree": 1.0, "reg_alpha": 0.0,
                                  "reg_lambda": 1.0}, X, y, seed=0)
    assert (model.predict(X) == "B").sum() >= 10


def test_logistic_regression_is_scaled():
    X, y = _data()
    model = fit_model("logistic_regression", _default_params("logistic_regression"), X, y, seed=0)
    assert model.estimator.steps[0][0] == "scale"


def test_sklearn_models_use_balanced_class_weight():
    X, y = _data()
    for name in ("random_forest", "lightgbm"):
        model = fit_model(name, _default_params(name), X, y, seed=0)
        assert model.estimator.get_params()["class_weight"] == "balanced"
    lr = fit_model("logistic_regression", _default_params("logistic_regression"), X, y, seed=0)
    assert lr.estimator.named_steps["model"].class_weight == "balanced"


def test_lightgbm_is_deterministic_across_thread_counts():
    # LightGBM only promises identical results across core counts with these set
    X, y = _data()
    params = fit_model("lightgbm", _default_params("lightgbm"), X, y, seed=0).estimator.get_params()
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True

