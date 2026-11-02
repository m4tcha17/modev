# Copra Moisture-Based Grade Classification

Image-based copra (dried coconut meat) grader. Each cut-face photo is assigned one of the buying station's six moisture grades, **Grade A–F** (F below 6%, A 6% to <9%, B 9% to <12%, C 12% to <15%, D 15% to <18%, E 18% and above; boundaries go to the higher grade, see [`docs/classes.md`](docs/classes.md)), from handcrafted image features (texture, color, edges) and a tuned tree-ensemble model. Classification only; no price is ever computed.

**Source of truth:** [`docs/model-development-instructions.md`](docs/model-development-instructions.md) (build spec) and [`docs/model-acceptance-criteria.md`](docs/model-acceptance-criteria.md) (pass/fail targets). Hard constraints: [`CLAUDE.md`](CLAUDE.md). Walkthrough: [`process.md`](process.md). Module map and known gaps: [`docs/architecture.md`](docs/architecture.md). Open parameters: [`docs/decisions/`](docs/decisions/).

## Status

**No model trained for reporting.** The full dataset (all six grades) is on the main PC, which is where feature extraction and training run. The copy in this repo's `dataset/` folder is an older, partial export used for development only. Check acceptance P1 (≥ 50 cubes per grade) on the full dataset.

Implemented and tested: data loading + cube checks, background model + GrabCut masking, lighting normalization, GLCM / HSV-LAB / Canny features, feature table with cached rotations/flips, outlier removal, StratifiedGroupKFold split + augmentation, four class-weighted models with Optuna, selection, basic metrics, background-only check.

Not yet matching the spec (see the gaps table in `docs/architecture.md`): nested tuning, acceptance pass/fail table, ablation, Brown-Duvel label check, TreeSHAP, artifact, Streamlit app, synthetic data generator.

Partial export in this repo (`dataset/copra_dataset/`): 576 photos, 144 cubes — Grade A 50, Grade B 44, Grade D 50. The background-only check fails on it, because each grade was photographed in its own session. Re-capture isn't possible, so this is accepted as a known limitation (ADR-011 for Grade D, ADR-012 for the background check).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 (`.python-version`) — newer versions break `llvmlite`'s build.

```bash
uv sync --extra dev
uv run pytest
```

## Run

```bash
uv run python scripts/extract_features.py   # -> data/processed/features.csv + mask_preview.jpg
uv run python scripts/prepare_folds.py      # outliers + folds -> features_clean.csv + folds.csv
uv run python scripts/background_check.py   # acceptance L2, before/after lighting normalization
uv run python scripts/train_models.py       # train/tune/select/evaluate -> data/processed/results/
```

Check `mask_preview.jpg` by eye after every new export: red outline = detected copra. Feature extraction and training are CPU-heavy.

## Project layout

```
src/copra_grading/
├── dataset.py           # load CSV + photos, cube checks
├── preprocessing/       # GrabCut masking, lighting normalization, resize
├── features/            # GLCM texture, HSV/LAB color, Canny edges
├── cleaning.py          # drop outlier rows
├── splitting.py         # StratifiedGroupKFold by batch_id (Cube ID), before augmentation
├── augmentation.py      # rotate/flip, training portions only
├── folds.py             # split, then pick augmented training copies
├── models/              # LR baseline, RF/XGBoost/LightGBM + Optuna
├── selection.py         # best ensemble on mean outer-fold Macro F1
├── evaluation.py        # metrics
├── background_check.py  # paper-only leakage check
├── explainability.py    # TreeSHAP by feature family
└── artifact.py          # model + config serialization

app/streamlit_app.py     # one photo in, one grade out (later effort)
configs/default.yaml     # all tunable parameters
scripts/                 # pipeline entry points
tests/                   # mirrors src/ layout
docs/                    # spec, acceptance criteria, architecture, ADRs
```

## Dataset layout

```
copra-dataset.csv      # id, batch_id, copra_class, path
photos/<id>.jpeg
```

One row per face photo. `batch_id` is the **Cube ID**: each cube has 4 photos (one per cut meat face) sharing it and the copra's grade. Each copra gives 1–3 cubes; there is no `copra_id`, so cubes of one copra can fall in different folds (a stated source of optimism). Fixed capture: 10 cm, locked camera settings, mid-grey paper, two white lights.

## Pipeline

1. **Load** — CSV + photos, keep `batch_id`; resolve duplicate Cube IDs.
2. **Preprocess** — background model + GrabCut mask → lighting normalization to a reference grey → resize.
3. **Features** — GLCM + HSV/LAB + Canny, one row per face photo.
4. **Outliers** — drop rows with any |z| > 3 or out-of-IQR-fence value; never impute or relax.
5. **Nested CV** — outer StratifiedGroupKFold(5) for reporting, inner (3) for Optuna on Macro F1; grouped by `batch_id`, stratified by grade; rotate/flip augmentation only after each split.
6. **Train** — Logistic Regression (standardized, baseline only), Random Forest, XGBoost, LightGBM; all class-weighted; never stacked.
7. **Select** — best of RF / XGBoost / LightGBM by mean outer-fold Macro F1.
8. **Evaluate** — per image on outer folds, against `docs/model-acceptance-criteria.md` (Macro F1 ≥ 0.75, every grade's recall ≥ 0.60, Grade A precision ≥ 0.85, gap over LR > 1 SD), plus background-only check, ablation, pooled Brown-Duvel label check.
9. **Explain** — TreeSHAP on the selected model, by feature family.
10. **Artifact** — retune and train the selected algorithm on all data; serialize with its config for the Streamlit app.
