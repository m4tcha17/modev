# Preprocessing + Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Otsu background masking (§4) and the three feature-extraction families — GLCM texture, HSV/LAB color, Canny edges (§5) — replacing the `NotImplementedError` stubs with real, tested code.

**Architecture:** Each angle image flows through one Otsu mask (`preprocessing/otsu.py`) producing a cropped, background-zeroed image, then through three independent feature extractors (`features/glcm.py`, `features/color.py`, `features/edges.py`) whose outputs `combine.py` concatenates into one ordered feature dict per angle image. This is the first of several pipeline-stage plans (cleaning/splitting/augmentation, models/tuning/selection, evaluation/artifact come later as their own plans) — this stage has no dependency on anything downstream and is fully testable in isolation with synthetic arrays.

**Tech Stack:** numpy, opencv-python (`cv2`), scikit-image (`skimage.filters.threshold_otsu`, `skimage.feature.graycomatrix`/`graycoprops`), pytest.

## Global Constraints

- Classification pipeline only — this stage produces feature vectors, not predictions; nothing here touches model code.
- No photometric augmentation anywhere (not relevant to this stage, but never introduce brightness/contrast/hue jitter into test fixtures as if it were a real pipeline step).
- Masked image (copra pixels only) is what every downstream feature extractor consumes — never the raw unmasked photo (§4).
- GLCM keeps each distance/angle combination as a separate feature, never averaged (ADR-006).
- Feature ordering must be deterministic — `artifact/serialize.py` (a later stage) will depend on reproducing the exact same ordering at inference time.
- **Never run `git commit`** — this repo's commits are done manually by the user (see `CLAUDE.md`). Steps below stage changes with `git add` only.

---

## File Structure

- `src/copra_grading/preprocessing/otsu.py` — replace stub. Adds `apply_mask()` alongside the existing `mask_image()` signature.
- `src/copra_grading/features/glcm.py` — replace stub.
- `src/copra_grading/features/color.py` — replace stub.
- `src/copra_grading/features/edges.py` — replace stub.
- `src/copra_grading/features/combine.py` — replace stub.
- `tests/preprocessing/test_otsu.py` — new.
- `tests/features/test_glcm.py` — new.
- `tests/features/test_color.py` — new.
- `tests/features/test_edges.py` — new.
- `tests/features/test_combine.py` — new.

---

### Task 1: Otsu background masking

**Files:**
- Modify: `src/copra_grading/preprocessing/otsu.py`
- Test: `tests/preprocessing/test_otsu.py`

**Interfaces:**
- Produces: `mask_image(image: np.ndarray) -> np.ndarray` — returns a 2D boolean array, same height/width as `image`, `True` = foreground (copra) pixel. Accepts grayscale or RGB (`H,W` or `H,W,3`) uint8 input.
- Produces: `apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray` — crops `image` to the bounding box of `mask`'s `True` pixels, zeroes out background pixels within that crop, returns the cropped array (same channel count as `image`). This cropped/zeroed array is what every downstream feature function calls "masked_image".
- Consumes: nothing from earlier tasks (first task in the plan).

- [ ] **Step 1: Write the failing tests**

```python
# tests/preprocessing/test_otsu.py
import numpy as np

from copra_grading.preprocessing.otsu import apply_mask, mask_image


def test_mask_image_isolates_bright_square_on_dark_background():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[15:25, 15:25] = 255

    mask = mask_image(image)

    assert mask.shape == (40, 40)
    assert mask.dtype == bool
    assert mask[20, 20]
    assert not mask[0, 0]
    assert mask.sum() == 100


def test_mask_image_isolates_dark_square_on_bright_background():
    image = np.full((40, 40), 200, dtype=np.uint8)
    image[15:25, 15:25] = 10

    mask = mask_image(image)

    assert mask[20, 20]
    assert not mask[0, 0]
    assert mask.sum() == 100


def test_apply_mask_crops_to_bounding_box_and_zeroes_background():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    # Cross shape: the bounding box's own corners fall outside the shape,
    # so the crop must still zero those corner pixels.
    mask[8:12, 2:18] = True
    mask[2:18, 8:12] = True
    image[mask] = [100, 150, 200]

    masked_image = apply_mask(image, mask)

    assert masked_image.shape == (16, 16, 3)
    assert list(masked_image[7, 7]) == [100, 150, 200]
    assert list(masked_image[0, 0]) == [0, 0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/preprocessing/test_otsu.py -v`
Expected: FAIL — `NotImplementedError` (and `ImportError: cannot import name 'apply_mask'` before Step 3 adds it).

- [ ] **Step 3: Implement**

```python
# src/copra_grading/preprocessing/otsu.py
"""Otsu background masking. Spec §4.

Primary/default masking method. Camera position is fixed across all six angle
shots of a sample (only the sample rotates), so one global threshold per image
is sufficient - no per-angle recalibration.

Fallback methods (adaptive threshold, HSV/saturation mask, GrabCut) are
documented in the spec as reach-for-if-needed, not preemptive - only implement
one if evaluation shows Otsu masks are unreliable under real field lighting.
Keep any fallback behind the same `mask_image` signature so feature extraction
never needs to know which method produced the mask.
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def mask_image(image: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating copra pixels from background.

    Steps (§4): grayscale conversion -> intensity histogram -> Otsu threshold
    -> binary mask. Downstream feature extraction must consume the masked
    image only, never the raw photo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    threshold = threshold_otsu(gray)
    above = gray > threshold
    below = ~above
    # Otsu splits pixels into two classes but doesn't say which one is the
    # object; the copra sample occupies less of the frame than the
    # tray/background behind it, so the minority-count side is foreground.
    return above if above.sum() < below.sum() else below


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop `image` to `mask`'s bounding box and zero out background pixels.

    This cropped, background-zeroed array is the "masked_image" every
    feature-extraction function downstream expects as input.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros_like(image)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_image = image[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1]
    cropped_image[~cropped_mask] = 0
    return cropped_image
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/preprocessing/test_otsu.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Stage changes**

```bash
git add src/copra_grading/preprocessing/otsu.py tests/preprocessing/test_otsu.py
```

---

### Task 2: GLCM texture features

**Files:**
- Modify: `src/copra_grading/features/glcm.py`
- Test: `tests/features/test_glcm.py`

**Interfaces:**
- Consumes: nothing directly (grayscale `np.ndarray`, e.g. from `cv2.cvtColor(apply_mask(...), cv2.COLOR_RGB2GRAY)` — that conversion happens in Task 5's `combine.py`, not here).
- Produces: `extract_glcm_features(masked_image: np.ndarray, distances: list[int], angles_deg: list[int]) -> dict[str, float]`. Keys are `f"glcm_d{d}_a{a}_{stat}"` for `stat in ("contrast", "homogeneity", "energy", "entropy")`, one set per `(distance, angle)` pair, outer loop over `distances` then `angles_deg` in the order given. `masked_image` must be a 2D uint8 grayscale array.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_glcm.py
import numpy as np

from copra_grading.features.glcm import extract_glcm_features


def test_constant_image_has_near_zero_contrast_and_entropy():
    gray = np.full((16, 16), 128, dtype=np.uint8)

    features = extract_glcm_features(gray, distances=[1], angles_deg=[0])

    assert features["glcm_d1_a0_contrast"] == 0.0
    assert features["glcm_d1_a0_entropy"] == 0.0
    assert features["glcm_d1_a0_energy"] == 1.0


def test_checkerboard_image_has_higher_contrast_than_constant_image():
    checkerboard = np.zeros((16, 16), dtype=np.uint8)
    checkerboard[::2, ::2] = 255
    checkerboard[1::2, 1::2] = 255
    constant = np.full((16, 16), 128, dtype=np.uint8)

    checker_features = extract_glcm_features(checkerboard, distances=[1], angles_deg=[0])
    constant_features = extract_glcm_features(constant, distances=[1], angles_deg=[0])

    assert checker_features["glcm_d1_a0_contrast"] > constant_features["glcm_d1_a0_contrast"]


def test_returns_one_set_of_stats_per_distance_angle_combination():
    gray = np.full((16, 16), 128, dtype=np.uint8)

    features = extract_glcm_features(gray, distances=[1, 2], angles_deg=[0, 90])

    expected_keys = {
        f"glcm_d{d}_a{a}_{stat}"
        for d in (1, 2)
        for a in (0, 90)
        for stat in ("contrast", "homogeneity", "energy", "entropy")
    }
    assert set(features.keys()) == expected_keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_glcm.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

```python
# src/copra_grading/features/glcm.py
"""GLCM texture features. Spec §5a. Distance/angle set: see ADR-006 / configs/default.yaml.

Physical rationale: wet copra has a smoother, more uniform surface; drying
roughens and cracks it. Contrast/homogeneity/energy/entropy quantify that.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    masked_image: np.ndarray,
    distances: list[int],
    angles_deg: list[int],
) -> dict[str, float]:
    """Return contrast, homogeneity, energy, entropy per distance/angle combo.

    Each distance/angle pair is kept as a separate feature (not averaged) per
    ADR-006's documented default.
    """
    gray = masked_image.astype(np.uint8)
    features: dict[str, float] = {}

    for d in distances:
        for a_deg in angles_deg:
            angle_rad = np.deg2rad(a_deg)
            glcm = graycomatrix(
                gray,
                distances=[d],
                angles=[angle_rad],
                levels=256,
                symmetric=True,
                normed=True,
            )
            probs = glcm[:, :, 0, 0]
            nonzero_probs = probs[probs > 0]
            entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

            # skimage's 'energy' prop is sqrt(ASM); the spec defines energy
            # as the sum of squared co-occurrence probabilities (ASM itself).
            asm = graycoprops(glcm, "ASM")[0, 0]

            key_prefix = f"glcm_d{d}_a{a_deg}"
            features[f"{key_prefix}_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
            features[f"{key_prefix}_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
            features[f"{key_prefix}_energy"] = float(asm)
            features[f"{key_prefix}_entropy"] = entropy

    return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_glcm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Stage changes**

```bash
git add src/copra_grading/features/glcm.py tests/features/test_glcm.py
```

---

### Task 3: HSV/LAB color features

**Files:**
- Modify: `src/copra_grading/features/color.py`
- Test: `tests/features/test_color.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extract_color_features(masked_image: np.ndarray) -> dict[str, float]`. `masked_image` is `H,W,3` uint8 RGB, background pixels exactly `[0,0,0]` (the `apply_mask` contract from Task 1). Keys are `f"{space}_{channel}_{stat}"` for `space in ("hsv","lab")`, `channel` in that space's three letters (`h,s,v` / `l,a,b`), `stat in ("mean","std")` — 12 keys total. Stats computed over foreground pixels only (any pixel where not all three channels are zero); if no foreground pixels exist, stats fall back to the whole image.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_color.py
import cv2
import numpy as np

from copra_grading.features.color import extract_color_features


def test_uniform_foreground_color_has_zero_std_and_matches_cv2_conversion():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_color_features(image)

    expected_hsv = cv2.cvtColor(np.uint8([[[200, 100, 50]]]), cv2.COLOR_RGB2HSV)[0, 0]
    expected_lab = cv2.cvtColor(np.uint8([[[200, 100, 50]]]), cv2.COLOR_RGB2LAB)[0, 0]

    assert features["hsv_h_std"] == 0.0
    assert features["hsv_s_std"] == 0.0
    assert features["hsv_v_std"] == 0.0
    assert features["hsv_h_mean"] == float(expected_hsv[0])
    assert features["lab_l_mean"] == float(expected_lab[0])


def test_returns_mean_and_std_for_all_six_channels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_color_features(image)

    expected_keys = {
        f"{space}_{channel}_{stat}"
        for space in ("hsv", "lab")
        for channel in space
        for stat in ("mean", "std")
    }
    assert set(features.keys()) == expected_keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_color.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

```python
# src/copra_grading/features/color.py
"""HSV/LAB color-space features. Spec §5b.

Physical rationale: moisture loss shifts copra from pale/white toward
tan/brown. HSV separates hue/saturation from brightness (less confounded by
uncontrolled ambient lighting than raw RGB); LAB is perceptually uniform and
matches the gray-to-brown drying shift.

Never apply photometric augmentation anywhere in this pipeline - it corrupts
this signal directly (see ../../../CLAUDE.md).
"""

import cv2
import numpy as np


def extract_color_features(masked_image: np.ndarray) -> dict[str, float]:
    """Return per-channel statistics (mean, std, and/or histogram stats) for HSV and LAB."""
    foreground_mask = np.any(masked_image != 0, axis=-1)
    if not foreground_mask.any():
        foreground_mask = np.ones(masked_image.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(masked_image, cv2.COLOR_RGB2LAB)

    features: dict[str, float] = {}
    for space_name, space_image in (("hsv", hsv), ("lab", lab)):
        for channel_idx, channel_name in enumerate(space_name):
            channel_values = space_image[..., channel_idx][foreground_mask].astype(np.float64)
            features[f"{space_name}_{channel_name}_mean"] = float(channel_values.mean())
            features[f"{space_name}_{channel_name}_std"] = float(channel_values.std())

    return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_color.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Stage changes**

```bash
git add src/copra_grading/features/color.py tests/features/test_color.py
```

---

### Task 4: Canny edge/contour density features

**Files:**
- Modify: `src/copra_grading/features/edges.py`
- Test: `tests/features/test_edges.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extract_edge_features(masked_image: np.ndarray, low_threshold: int, high_threshold: int) -> dict[str, float]`. `masked_image` may be grayscale (`H,W`) or RGB (`H,W,3`) — converts to grayscale internally if 3-channel. Returns `{"edge_density": float, "edge_contour_count": float}`. `edge_density` = edge-pixel count / foreground-pixel count (nonzero grayscale pixels), not raw image area, so bounding-box padding from `apply_mask` doesn't dilute it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_edges.py
import numpy as np

from copra_grading.features.edges import extract_edge_features


def test_uniform_image_has_zero_edge_density():
    gray = np.full((20, 20), 128, dtype=np.uint8)

    features = extract_edge_features(gray, low_threshold=50, high_threshold=150)

    assert features["edge_density"] == 0.0


def test_half_black_half_white_image_has_positive_edge_density():
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255

    features = extract_edge_features(gray, low_threshold=50, high_threshold=150)

    assert features["edge_density"] > 0.0
    assert features["edge_contour_count"] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_edges.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

```python
# src/copra_grading/features/edges.py
"""Canny edge/contour density features. Spec §5c.

Physical rationale: as copra dries and shrinks, its surface develops more
visible cracks/fissures. Edge density is a proxy for that structural change.
"""

import cv2
import numpy as np


def extract_edge_features(
    masked_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, float]:
    """Return edge-pixel proportion and/or contour distribution statistics."""
    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    foreground_pixel_count = int(np.count_nonzero(gray))
    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = edge_pixel_count / max(foreground_pixel_count, 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "edge_density": float(edge_density),
        "edge_contour_count": float(len(contours)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_edges.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Stage changes**

```bash
git add src/copra_grading/features/edges.py tests/features/test_edges.py
```

---

### Task 5: Combine features into one row per angle image

**Files:**
- Modify: `src/copra_grading/features/combine.py`
- Test: `tests/features/test_combine.py`

**Interfaces:**
- Consumes: `extract_glcm_features(masked_image, distances, angles_deg)` (Task 2), `extract_color_features(masked_image)` (Task 3), `extract_edge_features(masked_image, low_threshold, high_threshold)` (Task 4).
- Produces: `extract_all_features(masked_image: np.ndarray, config: dict) -> dict[str, float]`. `masked_image` is `H,W,3` uint8 RGB (the `apply_mask` output). `config` follows `configs/default.yaml`'s shape: `config["features"]["glcm"]["distances"]`, `config["features"]["glcm"]["angles_deg"]`, `config["features"]["canny"]["low_threshold"]`, `config["features"]["canny"]["high_threshold"]`. Grayscale conversion for GLCM/edges happens once here, not duplicated inside each extractor. Insertion order: GLCM keys, then color keys, then edge keys — this fixed order is what a later artifact-serialization stage relies on for deployment-time reproducibility.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_combine.py
import numpy as np

from copra_grading.features.combine import extract_all_features


def _config():
    return {
        "features": {
            "glcm": {"distances": [1], "angles_deg": [0, 90]},
            "canny": {"low_threshold": 50, "high_threshold": 150},
        }
    }


def test_combines_all_three_feature_families():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    features = extract_all_features(image, _config())

    assert any(key.startswith("glcm_") for key in features)
    assert any(key.startswith("hsv_") for key in features)
    assert any(key.startswith("lab_") for key in features)
    assert "edge_density" in features
    assert all(isinstance(value, float) for value in features.values())


def test_feature_ordering_is_deterministic_across_calls():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [200, 100, 50]

    first_call = list(extract_all_features(image, _config()).keys())
    second_call = list(extract_all_features(image, _config()).keys())

    assert first_call == second_call
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_combine.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

```python
# src/copra_grading/features/combine.py
"""Concatenate GLCM + color + edge features into one row per angle image. Spec §5."""

import cv2
import numpy as np

from copra_grading.features.color import extract_color_features
from copra_grading.features.edges import extract_edge_features
from copra_grading.features.glcm import extract_glcm_features


def extract_all_features(masked_image: np.ndarray, config: dict) -> dict[str, float]:
    """Run all three feature families and concatenate into one dict.

    Feature ordering here must match whatever ordering artifact/serialize.py
    records - deployment-time extraction has to reproduce it exactly.
    """
    gray = (
        cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        if masked_image.ndim == 3
        else masked_image
    )

    glcm_config = config["features"]["glcm"]
    canny_config = config["features"]["canny"]

    features: dict[str, float] = {}
    features.update(
        extract_glcm_features(
            gray,
            distances=glcm_config["distances"],
            angles_deg=glcm_config["angles_deg"],
        )
    )
    features.update(extract_color_features(masked_image))
    features.update(
        extract_edge_features(
            gray,
            low_threshold=canny_config["low_threshold"],
            high_threshold=canny_config["high_threshold"],
        )
    )
    return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_combine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: PASS — all tests from this plan plus the pre-existing `test_labels.py`.

- [ ] **Step 6: Stage changes**

```bash
git add src/copra_grading/features/combine.py tests/features/test_combine.py
```

---

## Self-Review Notes

- **Spec coverage:** §4 (Otsu masking) → Task 1. §5a (GLCM) → Task 2. §5b (HSV/LAB) → Task 3. §5c (Canny) → Task 4. §5 combined output → Task 5. ADR-006 (GLCM distance/angle set kept separate, not averaged) → Task 2's per-combo key naming.
- **Not covered by this plan (future plans):** cleaning/dedup, cleaning/outliers, splitting/groupkfold, augmentation/geometric, models/*, selection/*, explainability/treeshap, evaluation/*, artifact/serialize — each depends on this stage's output shape and gets its own plan once this one lands.
- **Type consistency check:** `apply_mask` output (cropped RGB, background zeroed) is exactly what `combine.py`'s `masked_image` parameter expects, and what `extract_color_features` assumes when building its foreground mask via `np.any(masked_image != 0, axis=-1)`. `extract_glcm_features` and `extract_edge_features` both take grayscale — `combine.py` does that conversion once and passes the same `gray` array to both, never passing RGB to either.
