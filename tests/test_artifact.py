import json

import pytest

from copra_grading.artifact import load_artifact, save_artifact
from copra_grading.models import fit_model

CONFIG = {"preprocessing": {"resize_to": [64, 48]}, "features": {"glcm": {}}, "tuning": {"n_trials": 3}}


@pytest.mark.parametrize("name, params", [
    ("random_forest", {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1}),
    ("xgboost", {"n_estimators": 20, "max_depth": 3}),
    ("lightgbm", {"n_estimators": 20, "max_depth": 3}),
])
def test_round_trip_gives_identical_predictions(tmp_path, feature_table, name, params):
    originals = feature_table[feature_table["augment"] == "rot0"]
    X, y = originals[["feat_1", "feat_2"]], originals["copra_class"]
    model = fit_model(name, params, X, y, seed=0)

    save_artifact(model, ["feat_1", "feat_2"], CONFIG, tmp_path, params=params, extra={"note": 1})
    loaded = load_artifact(tmp_path)

    assert list(loaded.predict(X[["feat_2", "feat_1"]])) == list(model.predict(X))  # column order restored
    assert loaded.model.name == name and list(loaded.model.classes) == list("ABCDE")
    assert loaded.config == {k: CONFIG[k] for k in ("preprocessing", "features")}  # no training-only sections
    assert loaded.params == params and loaded.extra == {"note": 1}


def test_xgboost_uses_native_format(tmp_path, feature_table):
    originals = feature_table[feature_table["augment"] == "rot0"]
    model = fit_model("xgboost", {"n_estimators": 5}, originals[["feat_1", "feat_2"]], originals["copra_class"], 0)
    save_artifact(model, ["feat_1", "feat_2"], CONFIG, tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["model_file"] == "model.json"


def test_missing_model_file_refuses_to_load(tmp_path, feature_table):
    originals = feature_table[feature_table["augment"] == "rot0"]
    model = fit_model("random_forest", {"n_estimators": 5}, originals[["feat_1", "feat_2"]], originals["copra_class"], 0)
    save_artifact(model, ["feat_1", "feat_2"], CONFIG, tmp_path)
    (tmp_path / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path)


def test_missing_directory_refuses_to_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nothing")
