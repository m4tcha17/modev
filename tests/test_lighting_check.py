import numpy as np
import pandas as pd

from copra_grading.lighting_check import grade_accuracy_from_paper, paper_features

CONFIG = {"splitting": {"n_splits": 5, "group_column": "batch_id",
                        "label_column": "copra_class", "random_state": 42}}


def _photo(paper):
    image = np.full((240, 320, 3), paper, np.uint8)
    image[80:160, 100:220] = (210, 205, 190)  # copra, outside the border band
    return image


def test_paper_features_read_the_paper_not_the_copra():
    f = paper_features(_photo(120))
    assert set(f) == {f"paper_{c}_{s}" for c in "lab" for s in ("mean", "std")}
    assert abs(f["paper_l_mean"] - 120 * 255 / 255) < 10  # neutral grey: L tracks the grey level
    assert f["paper_l_std"] < 1


def test_normalization_equalizes_paper_brightness():
    dim, bright = paper_features(_photo(90), 110), paper_features(_photo(170), 110)
    assert abs(dim["paper_l_mean"] - bright["paper_l_mean"]) < 2
    raw = abs(paper_features(_photo(90))["paper_l_mean"] - paper_features(_photo(170))["paper_l_mean"])
    assert raw > 50


def _table(leaky):
    rng = np.random.default_rng(0)
    rows = []
    for b in range(50):
        cls = "ABCDE"[b % 5]
        for p in range(4):
            signal = "ABCDE".index(cls) * 10 if leaky else 0
            rows.append({"batch_id": f"b{b}", "copra_class": cls,
                         "paper_l_mean": signal + rng.normal(0, 1), "paper_l_std": rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_paper_that_tracks_the_grade_is_a_leak():
    assert grade_accuracy_from_paper(_table(True), CONFIG)["mean_accuracy"] > 0.9


def test_uninformative_paper_stays_near_chance():
    result = grade_accuracy_from_paper(_table(False), CONFIG)
    assert result["chance"] == 0.2 and result["n_classes"] == 5
    assert result["mean_accuracy"] < 0.35
