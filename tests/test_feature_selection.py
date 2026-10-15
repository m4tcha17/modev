import numpy as np
import pandas as pd

from copra_grading.feature_selection import select_uncorrelated, selected_features


def _table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(n)],
        "batch_id": [f"b{i}" for i in range(n)],
        "copra_class": ["A"] * n,
        "augment": ["rot0"] * n,
        "mask_fraction": 0.3,
        "feat_a": a,
        "feat_a_copy": a * 2 + 1,           # perfectly correlated with feat_a
        "feat_a_rank": np.exp(a),           # monotone in feat_a: Spearman 1
        "feat_b": b,
    })


def test_drops_features_correlated_with_an_earlier_one():
    table = _table()
    kept = select_uncorrelated(table, ["feat_a", "feat_a_copy", "feat_a_rank", "feat_b"], 0.95)
    assert kept == ["feat_a", "feat_b"]


def test_keeps_first_in_extraction_order():
    table = _table()
    kept = select_uncorrelated(table, ["feat_a_copy", "feat_a", "feat_b"], 0.95)
    assert kept == ["feat_a_copy", "feat_b"]


def test_uses_original_rows_only():
    table = _table()
    # augmented rows that would make feat_b track feat_a must be ignored
    aug = table.copy()
    aug["augment"] = "rot90"
    aug["feat_b"] = aug["feat_a"] * 1000
    both = pd.concat([table, pd.concat([aug] * 20)], ignore_index=True)
    assert select_uncorrelated(both, ["feat_a", "feat_b"], 0.95) == ["feat_a", "feat_b"]


def test_none_threshold_keeps_every_feature(config):
    table = _table()
    config["feature_selection"] = {"correlation_threshold": None}
    assert selected_features(table, config) == ["feat_a", "feat_a_copy", "feat_a_rank", "feat_b"]


def test_missing_config_section_keeps_every_feature(config):
    table = _table()
    assert "feature_selection" not in config
    assert len(selected_features(table, config)) == 4


def test_config_threshold_is_applied(config):
    config["feature_selection"] = {"correlation_threshold": 0.95}
    assert selected_features(_table(), config) == ["feat_a", "feat_b"]
