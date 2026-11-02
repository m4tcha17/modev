# app/functions.py

##from app.backend.model_loader import load_model_artifact
## from app.backend.prediction import predict_image
##from backend.explanation import explain_prediction
from backend.prediction import extract_image_features
from pathlib import Path
import yaml
from dataclasses import dataclass
import pandas as pd
import numpy as np
from backend.prediction import extract_image_features

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the existing research pipeline configuration
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)


@dataclass
class ProcessingResult:
    label: str
    grade: str
    status_label: str
    features: pd.DataFrame
    mask: np.ndarray


def process_image(image):
    features, mask = extract_image_features(image, CONFIG)

    return ProcessingResult(
        label="Class A",
        grade="Low Moisture / High Quality",
        status_label="MOCK CLASSIFICATION",
        features=features,
        mask=mask,
    )

def explain_prediction(result, artifact):
    return explain_prediction(
        model=artifact["model"],
        features=result["features"],
        predicted_grade=result["grade"],
    )