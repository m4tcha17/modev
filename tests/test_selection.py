import pytest

import copra_grading.selection as selection
from copra_grading.selection import cross_validate_models, select_model


def test_select_model_ignores_logistic_regression():
    scores = {"logistic_regression": [0.99] * 5, "random_forest": [0.7] * 5,
              "xgboost": [0.8] * 5, "lightgbm": [0.75] * 5}
    assert select_model(scores) == "xgboost"


def test_select_model_needs_an_ensemble():
    with pytest.raises(ValueError, match="ensemble"):
        select_model({"logistic_regression": [0.9] * 5})


def test_every_photo_predicted_once_per_model(feature_table, config):
    result = cross_validate_models(
        feature_table, config, ("logistic_regression", "random_forest"), n_trials=1
    )
    originals = feature_table[feature_table["augment"] == "rot0"]
    for name in ("logistic_regression", "random_forest"):
        preds = result.predictions[result.predictions["model"] == name]
        assert sorted(preds["id"]) == sorted(originals["id"])
        assert len(result.scores[name]) == 5
        assert len(result.best_params[name]) == 5
    assert set(result.predictions["predicted"]) <= set("ABCDE")


def test_tuning_never_sees_outer_validation_photos(feature_table, config, monkeypatch):
    seen = []
    real = selection.tune_model

    def spy(name, table, cfg, n_trials, seed):
        seen.append(table)
        return real(name, table, cfg, n_trials, seed)

    monkeypatch.setattr(selection, "tune_model", spy)
    result = cross_validate_models(feature_table, config, ("random_forest",), n_trials=1)
    preds = result.predictions
    for k, tuned in enumerate(seen):
        val = preds[preds["fold"] == k]
        assert set(tuned["id"]).isdisjoint(val["id"])
        assert set(tuned["batch_id"]).isdisjoint(val["batch_id"])
        # tuning gets every transform of its photos, so inner folds can augment
        assert len(tuned) == 8 * tuned["id"].nunique()

