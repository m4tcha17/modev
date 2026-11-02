"""Background-only check. Thesis spec section 11, target T5 (goal.md).

Train a simple classifier on paper-only pixels (no copra) to predict the
grade. Accuracy near chance means lighting does not reveal the grade;
clearly above chance means it leaks grade information and model scores
can't be trusted. Run before and after lighting normalization.
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from copra_grading.splitting import split_by_batch

_WORK_WIDTH = 640  # paper stats don't need full resolution
_MARGIN = 0.03  # paper = outside the copra mask dilated by this share of the width


def paper_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB pixels of the paper only: everything outside a dilated copra mask."""
    h0, w0 = image.shape[:2]
    w = min(_WORK_WIDTH, w0)
    h = max(1, round(h0 * w / w0))
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, round(_MARGIN * w) | 1)
    copra = cv2.dilate(small_mask, np.ones((k, k), np.uint8)).astype(bool)
    return small[~copra]


def paper_features(pixels: np.ndarray) -> dict:
    """Mean and SD of each RGB and LAB channel over the paper pixels."""
    if len(pixels) == 0:
        raise ValueError("no paper pixels")
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rgb = pixels.astype(np.float32)
    out = {}
    for space, arr, names in (("rgb", rgb, "rgb"), ("lab", lab, ("L", "a", "b"))):
        for i, ch in enumerate(names):
            out[f"paper_{space}_{ch}_mean"] = float(arr[:, i].mean())
            out[f"paper_{space}_{ch}_std"] = float(arr[:, i].std())
    return out


def classify_from_paper(
    table: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    group_column: str = "batch_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Grouped, stratified CV of two simple classifiers predicting
    `copra_class` from paper features. One row per (model, fold)."""
    models = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=2
        ),
    }
    table = table.reset_index(drop=True)
    splits = split_by_batch(table, n_splits, group_column, "copra_class", random_state)
    rows = []
    for name, make in models.items():
        for k, (tr, va) in enumerate(splits):
            model = make().fit(table.loc[tr, feature_cols], table.loc[tr, "copra_class"])
            pred = model.predict(table.loc[va, feature_cols])
            y = table.loc[va, "copra_class"]
            rows.append({"model": name, "fold": k, "accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, average="macro")})
    return pd.DataFrame(rows)
