import numpy as np
import pandas as pd
import pytest

from copra_grading.explainability import (
    FAMILIES,
    compute_aggregate_shap_by_family,
    explain_single_prediction,
    family_shares,
    feature_family,
    feature_family_map,
)
from copra_grading.models import fit_model


def test_feature_family_from_prefix():
    assert feature_family("glcm_d1_a0_contrast") == "texture"
    assert feature_family("hsv_h_mean") == "color" and feature_family("lab_l_std") == "color"
    assert feature_family("edge_density") == "edge"
    with pytest.raises(ValueError):
        feature_family("mask_fraction")


@pytest.fixture
def data():
    """Texture separates the classes; color and edge are noise."""
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(list("ABC"), 60))
    centre = y.map({"A": 0.0, "B": 3.0, "C": 6.0}).to_numpy()
    X = pd.DataFrame({
        "glcm_contrast": centre + rng.normal(0, 0.3, len(y)),
        "hsv_h_mean": rng.normal(0, 1, len(y)),
        "edge_density": rng.normal(0, 1, len(y)),
    })
    return X, y


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "lightgbm"])
def test_aggregate_ranks_the_informative_family_first(data, name):
    X, y = data
    model = fit_model(name, {"n_estimators": 30, "max_depth": 3}, X, y, seed=0)
    result = compute_aggregate_shap_by_family(model, X, feature_family_map(X.columns))
    assert set(result) == set(FAMILIES)
    assert all(v >= 0 for v in result.values())
    assert max(result, key=result.get) == "texture"


def test_single_prediction_gives_every_family(data):
    X, y = data
    model = fit_model("lightgbm", {"n_estimators": 30, "max_depth": 3}, X, y, seed=0)
    contributions = explain_single_prediction(model, X.iloc[0], feature_family_map(X.columns))
    assert set(contributions) == set(FAMILIES)
    assert contributions["texture"] > 0  # texture pushes toward the (correct) predicted class
    shares = family_shares(contributions)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_family_shares_of_all_zero():
    assert family_shares({"texture": 0.0, "color": 0.0, "edge": 0.0}) == {"texture": 0.0, "color": 0.0, "edge": 0.0}
