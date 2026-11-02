# app/backend/prediction.py

import pandas as pd

from copra_grading.preprocessing.pipeline import compute_mask
from copra_grading.preprocessing.otsu import apply_mask, resize_masked
from copra_grading.features.combine import extract_all_features
from copra_grading.features import EmptyRegionError


def extract_image_features(image, config):
    """
    Apply the same preprocessing and feature extraction used during training.

    Args:
        image: RGB image as a NumPy array.
        config: Project configuration dictionary.

    Returns:
        features: One-row DataFrame containing extracted features.
        mask: Binary copra mask.
    """
    mask = compute_mask(image, config)

    if mask is None or not mask.any():
        raise ValueError("No copra region detected in the uploaded image.")

    crop = apply_mask(image, mask)

    size = config["preprocessing"]["resize_to"]
    processed_image = resize_masked(crop, size)

    try:
        feature_values = extract_all_features(processed_image, config)
    except EmptyRegionError as err:
        raise ValueError(
            f"Unable to extract features from the uploaded image: {err}"
        ) from err

    features = pd.DataFrame([feature_values])

    return features, mask


def predict_image(image, artifact, config):
    """
    Predict the copra grade from one uploaded image.

    Args:
        image: RGB image as a NumPy array.
        artifact: Loaded model artifact containing 'model' and 'feature_columns'.
        config: Project configuration dictionary.

    Returns:
        Dictionary containing the predicted grade, mask, and extracted features.
    """
    features, mask = extract_image_features(image, config)

    expected_columns = artifact["feature_columns"]

    missing_columns = [
        column for column in expected_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Extracted features do not match the trained model. "
            f"Missing columns: {missing_columns}"
        )

    features = features[expected_columns]

    prediction = artifact["model"].predict(features)

    return {
        "grade": str(prediction[0]),
        "mask": mask,
        "features": features,
    }