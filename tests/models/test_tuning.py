from copra_grading.models.tuning import inner_config, tune_model


def test_returns_study_with_macro_f1_value(feature_table, config):
    study = tune_model("random_forest", feature_table, config, n_trials=2, seed=0)
    assert len(study.trials) == 2
    assert 0.0 <= study.best_value <= 1.0
    assert {"max_depth", "min_samples_leaf"} <= set(study.best_params)


def test_same_seed_same_best_params(feature_table, config):
    a = tune_model("lightgbm", feature_table, config, n_trials=3, seed=7)
    b = tune_model("lightgbm", feature_table, config, n_trials=3, seed=7)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value


def test_inner_config_uses_inner_split_count(config):
    inner = inner_config(config)
    assert inner["splitting"]["n_splits"] == 3
    assert config["splitting"]["n_splits"] == 5  # caller's config untouched
