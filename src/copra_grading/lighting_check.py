"""Background-only lighting check. Thesis §4.10, REQ-4.1-8; ADR-013 item 2.

Question: does the lighting alone reveal the grade? Grades were partly
photographed in sequence while the capture lights dimmed, so brightness can
line up with grade. A classifier is trained on paper-only pixels - the
border band, no copra in it - to predict the grade, with the same grouped,
stratified folds the models use. Accuracy near chance (1 / number of
classes) means the paper carries no grade information; clearly above
chance means lighting leaks grade information into the photos.

It runs on the raw photos and again after grey-paper normalization
(normalization.py). Caveat: normalization scales each photo so its paper
average hits the reference grey, so after it the paper means are nearly
constant by construction. What the "after" run still measures is everything
the gain does not remove: spread, color balance across channels, and clipping.
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from copra_grading.preprocessing.background import border_band_mask, working_copy
from copra_grading.preprocessing.normalization import apply_gains, lighting_gains
from copra_grading.splitting import split_by_batch


def paper_features(image: np.ndarray, reference_grey: float | None = None) -> dict[str, float]:
    """Mean and SD of L, a, b over the border band of one RGB photo.

    With `reference_grey`, the photo is lighting-normalized first (gains from
    the full photo, applied to the downscaled copy the band is read from).
    """
    small, border_band = working_copy(image)
    if reference_grey is not None:
        small = apply_gains(small, lighting_gains(image, reference_grey))
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    band = lab[border_band_mask(lab.shape[:2], border_band)].astype(np.float64)
    features = {}
    for i, channel in enumerate("lab"):
        features[f"paper_{channel}_mean"] = float(band[:, i].mean())
        features[f"paper_{channel}_std"] = float(band[:, i].std())
    return features


def grade_accuracy_from_paper(table: pd.DataFrame, config: dict) -> dict:
    """Grouped, stratified CV accuracy of a class-weighted Logistic Regression
    predicting the grade from paper features only.

    `table` has the split's group and label columns plus `paper_*` columns.
    """
    split = config["splitting"]
    label = split["label_column"]
    columns = [c for c in table.columns if c.startswith("paper_")]
    folds = split_by_batch(table, split["n_splits"], split["group_column"], label, split["random_state"])
    accuracies = []
    for train_idx, val_idx in folds:
        train, val = table.iloc[train_idx], table.iloc[val_idx]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=2000))
        model.fit(train[columns], train[label])
        accuracies.append(float(accuracy_score(val[label], model.predict(val[columns]))))
    n_classes = table[label].nunique()
    return {
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "fold_accuracy": [round(a, 4) for a in accuracies],
        "chance": round(1 / n_classes, 4),
        "n_classes": int(n_classes),
        "n_photos": int(len(table)),
    }
