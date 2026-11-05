import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from copra_grading.artifact import load_artifact, save_artifact
from copra_grading.features.combine import extract_all_features
from copra_grading.inference import MAX_UPLOAD_BYTES, UploadError, classify, decode_upload, explain
from copra_grading.models import fit_model
from copra_grading.preprocessing.pipeline import preprocess

ROOT = Path(__file__).resolve().parents[1]


def _png(image):
    return cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()


def _photo(meat, paper=140, seed=0):
    """A grey-paper photo with one copra cube; `meat` sets the cut face color."""
    rng = np.random.default_rng(seed)
    image = np.full((240, 320, 3), paper, np.float32) + rng.normal(0, 2, (240, 320, 3))
    image[70:170, 110:210] = meat
    image[70:170, 110:210] += rng.normal(0, 6, (100, 100, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    config = copy.deepcopy(config)
    config["preprocessing"]["resize_to"] = [96, 96]
    rows, labels = [], []
    for i in range(12):
        for cls, meat in (("A", (235, 230, 215)), ("E", (150, 120, 90))):
            image = _photo(meat, paper=120 + 4 * i, seed=i)
            rows.append(extract_all_features(preprocess(image, config), config))
            labels.append(cls)
    X = pd.DataFrame(rows)
    model = fit_model("random_forest", {"n_estimators": 30, "max_depth": 4}, X, pd.Series(labels), seed=0)
    out = tmp_path_factory.mktemp("artifact")
    save_artifact(model, list(X.columns), config, out)
    return load_artifact(out)


def test_decode_accepts_png_and_jpeg():
    image = _photo((230, 225, 210))
    assert decode_upload(_png(image)).shape == image.shape
    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
    assert decode_upload(jpeg).shape == image.shape


@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"GIF89a" + b"\0" * 100, "not a JPEG or PNG"),
    (b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, "corrupted"),
    (b"\xff\xd8\xff" + b"\0" * (MAX_UPLOAD_BYTES), "limit"),
], ids=["empty", "gif", "corrupt-png", "too-large"])
def test_decode_rejects_bad_files(data, message):
    with pytest.raises(UploadError, match=message):
        decode_upload(data)


def test_extension_is_not_trusted():
    # A text file renamed .png still fails the signature check.
    with pytest.raises(UploadError, match="not a JPEG or PNG"):
        decode_upload(b"hello, I am a text file")


def test_classify_returns_a_grade_from_the_model(artifact):
    white = classify(artifact, decode_upload(_png(_photo((235, 230, 215), paper=150, seed=99))))
    brown = classify(artifact, decode_upload(_png(_photo((150, 120, 90), paper=130, seed=98))))
    assert white.grade == "A" and brown.grade == "E"
    assert white.description.startswith("6-9%")
    assert white.mask.shape == (240, 320) and white.mask.any()


def test_classify_rejects_a_photo_without_copra(artifact):
    blank = np.full((240, 320, 3), 140, np.uint8)
    with pytest.raises(UploadError, match="No copra"):
        classify(artifact, blank)


def test_explain_gives_family_shares(artifact):
    result = classify(artifact, _photo((235, 230, 215), seed=5))
    shares = explain(artifact, result.features)
    assert set(shares) == {"texture", "color", "edge"}
    assert sum(shares.values()) == pytest.approx(1.0)
