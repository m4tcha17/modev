import numpy as np
import pandas as pd
import pytest

from copra_grading.augmentation import TRANSFORMS

CONFIG = {
    "splitting": {"n_splits": 5, "group_column": "batch_id",
                  "label_column": "copra_class", "random_state": 42},
    "augmentation": {"base_copies": 1, "max_multiplier": 4},
    "tuning": {"n_trials": 2, "inner_n_splits": 3, "random_state": 42,
               "objective_metric": "f1_macro"},
}


@pytest.fixture
def config():
    return {k: dict(v) for k, v in CONFIG.items()}


@pytest.fixture
def feature_table():
    """Feature table shaped like features_clean.csv: 40 batches x 4 photos x
    8 transforms, classes A-E (no F), two informative features."""
    rng = np.random.default_rng(0)
    classes = ("A", "B", "C", "D", "E")
    rows = []
    for b in range(40):
        cls = classes[b % 5]
        centre = classes.index(cls) * 3.0
        for p in range(4):
            base = centre + rng.normal(0, 0.3, size=2)
            for t in TRANSFORMS:
                f1, f2 = base + rng.normal(0, 0.05, size=2)
                rows.append({"id": f"B{b}-{p}", "batch_id": f"B{b}", "copra_class": cls,
                             "augment": t, "mask_fraction": 0.3,
                             "feat_1": f1, "feat_2": f2})
    return pd.DataFrame(rows)
