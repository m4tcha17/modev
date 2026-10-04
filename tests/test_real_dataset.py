"""Checks against the real export in dataset/. Skipped when it is absent
(dataset/ is gitignored)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from copra_grading.dataset import check_batches, load_dataset, load_image
from copra_grading.features.combine import extract_all_features
from copra_grading.preprocessing.otsu import apply_mask, resize_masked
from copra_grading.preprocessing.pipeline import compute_mask

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
CSV = ROOT / CONFIG["data"]["csv_path"]

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="real dataset not present")


@pytest.fixture(scope="module")
def df():
    return load_dataset(CSV)


def test_real_export_matches_schema_and_batches(df):
    assert check_batches(df).is_clean


@pytest.mark.parametrize("pos", [0, 0.25, 0.5, 0.75, 1.0])
def test_real_photo_mask_is_plausible(df, pos):
    # Observed on the 2026-10-04 export: copra covers 13-37% of the frame
    # and sits near the centre. Otsu gave ~77% (background included).
    row = df.iloc[int(pos * (len(df) - 1))]
    image = load_image(CSV, row["path"])

    mask = compute_mask(image, CONFIG)

    assert 0.05 < mask.mean() < 0.5
    border = np.concatenate([mask[:20].ravel(), mask[-20:].ravel(), mask[:, :20].ravel(), mask[:, -20:].ravel()])
    assert not border.any()

    masked = resize_masked(apply_mask(image, mask), CONFIG["preprocessing"]["resize_to"])
    feats = extract_all_features(masked, CONFIG)
    assert np.isfinite(list(feats.values())).all()
