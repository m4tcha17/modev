# Copra Moisture-Based Grade Classification

Image-based quality grading pipeline for copra (dried coconut meat). Classifies a copra sample into one of three PCA AO 02 moisture brackets — **Class 1** (≤6.0% MC, premium), **Class 2** (6.1–13.9% MC, graduated deductions), **Class 3** (≥14.0% MC, rejected) — from handcrafted image features (texture, color, edge density) and a tuned tree-ensemble model. Classification only; no price prediction.

Full build spec: [`model-development-instructions.md`](model-development-instructions.md). Hard constraints: [`CLAUDE.md`](CLAUDE.md). Design detail: [`docs/architecture.md`](docs/architecture.md). Unresolved parameters: [`docs/decisions/`](docs/decisions/).

## Status

**Scaffold stage — no trained model yet.** `src/copra_grading/` contains the full module structure with documented interfaces; most functions currently `raise NotImplementedError`. The one real implementation is `labels.py` (class-label derivation from moisture reading), which is tested. Everything else — masking, feature extraction, cleaning, splitting, augmentation, training, selection, explainability, evaluation, serialization — is pending implementation against a real or synthetic dataset.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 (`.python-version`) — newer versions break `llvmlite`'s build.

```bash
uv sync --extra dev
uv run pytest
```

## Project layout

```
src/copra_grading/
├── labels.py            # class-label derivation (implemented)
├── preprocessing/        # Otsu background masking
├── features/              # GLCM texture, HSV/LAB color, Canny edges
├── cleaning.py             # duplicate resolution, outlier detection
├── splitting.py             # GroupKFold by Sample_ID (before augmentation)
├── augmentation.py           # geometric-only, class-weighted
├── models/                    # Logistic Regression baseline, RF/XGBoost/LightGBM + Optuna tuning
├── selection.py                 # algorithm selection vs. deployment-angle configuration
├── explainability.py              # TreeSHAP, aggregated by feature family
├── evaluation.py                   # metrics, boundary analysis, human baseline, ablation
└── artifact.py                      # model + config serialization

configs/default.yaml   # all tunable parameters in one place
notebooks/              # exploration only — no pipeline logic
scripts/                 # synthetic dataset generator (for dev before real data lands)
tests/                    # mirrors src/ layout
docs/                      # architecture doc + ADRs for unresolved spec parameters
```

## Pipeline, end to end

1. **Input** — a CSV export (`Sample_ID`, `Angle_ID`, `timestamp`, `moisture_reading`) plus six angle images per physical sample, referenced via a configurable image root (`configs/default.yaml: data.image_root`).
2. **Masking** — Otsu thresholding isolates copra pixels from background per image.
3. **Feature extraction** — GLCM texture + HSV/LAB color + Canny edge density, concatenated per angle image.
4. **Cleaning** — duplicate `Sample_ID` conflicts are detected and reported (a `DedupReport`), not removed; then outlier flagging on extracted features (advisory only; nothing is dropped). Outlier detection defaults to IQR with a 1.5x multiplier and flags a row only when >=3 features are out of bounds (`cleaning.outlier_min_features`). Method, multiplier, and threshold are ADR-001 defaults the thesis team has not yet confirmed against real feature distributions.
5. **Split → augment** — GroupKFold by `Sample_ID` first, then geometric-only augmentation with per-class multipliers (never the reverse — see `CLAUDE.md`).
6. **Training + tuning** — Logistic Regression baseline plus Random Forest, XGBoost, LightGBM, all class-weighted, tuned with Optuna for Macro F1.
7. **Algorithm selection** — best-scoring ensemble on the combined all-angle feature set is selected; Logistic Regression never deploys.
8. **Deployment configuration** — the selected algorithm is retrained on single-angle / angle-subset feature sets to find the minimal photo input a live user needs to submit.
9. **Explainability** — TreeSHAP on the selected algorithm's combined all-angle version, aggregated by feature family.
10. **Evaluation** — confusion matrix, per-class precision/recall (Class 1 precision and Class 3 recall emphasized), Macro F1, boundary-region vs. mid-range breakdown, human-baseline comparison, feature-family ablation.
11. **Artifact** — the angle-retrained deployment model plus its exact preprocessing/feature config is serialized. This is the only artifact handed off downstream — never the combined all-angle model from step 7.

## How the trained model will be used (once implemented)

Once `artifact.py` produces a real artifact, downstream usage is:

```python
from copra_grading.artifact import load_artifact

model, config = load_artifact("path/to/artifact")

# For a single submitted photo:
# 1. mask it with the same Otsu params in `config`
# 2. extract features with `features/combine.py`, using `config`'s feature ordering
# 3. model.predict(...) -> "1" | "2" | "3"
# 4. optional: explainability.explain_single_prediction(model, features) for a live SHAP explanation
```

This is exactly what the separate Streamlit deployment interface (out of scope for this repo) will call — one submitted photo in, one class out, no price adjustment computed anywhere in this system.

## Developing against synthetic data

If real field data isn't available yet:

```bash
uv run python scripts/make_synthetic_dataset.py --n-samples 200 --out-dir data/raw
```

Generates a schema-matching placeholder dataset so the pipeline can be built and tested end-to-end before real data lands.
