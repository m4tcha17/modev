import pytest

import copra_grading.selection as selection
from copra_grading.selection import fit_final_model, full_training_rows


def test_full_training_rows_keep_every_photo_and_add_copies(feature_table, config):
    rows = full_training_rows(feature_table, config)
    originals = feature_table[feature_table["augment"] == "rot0"]
    assert set(rows["id"]) == set(originals["id"])
    assert (rows["augment"] == "rot0").sum() == len(originals)
    assert len(rows) > len(originals)


def test_final_fit_retunes_on_all_photos_with_five_folds(feature_table, config, monkeypatch):
    calls = []
    real = selection.tune_model

    def spy(name, table, cfg, n_trials, seed, n_splits=None):
        calls.append((name, table, n_splits))
        return real(name, table, cfg, n_trials, seed, n_splits=n_splits)

    monkeypatch.setattr(selection, "tune_model", spy)
    model, params = fit_final_model(feature_table, config, "random_forest", n_trials=1)

    [(name, table, n_splits)] = calls
    assert name == "random_forest" and n_splits == 5 and len(table) == len(feature_table)
    assert set(model.predict(feature_table[["feat_1", "feat_2"]])) <= set("ABCDE")
    assert isinstance(params, dict)


def test_logistic_regression_is_never_deployed(feature_table, config):
    with pytest.raises(ValueError, match="not deployable"):
        fit_final_model(feature_table, config, "logistic_regression", n_trials=1)
