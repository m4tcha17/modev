# Copra Classification

Image-based copra (dried coconut meat) classifier. Each photo is assigned one of six moisture-content classes, **A–F** (F below 6%, A 6–9%, B 9–12%, C 12–15%, D 15–18%, E 18%+; see [`docs/classes.md`](docs/classes.md)), from handcrafted image features (texture, color, edges) and a tuned tree-ensemble model.

Build spec: [`process.md`](process.md). Hard constraints: [`CLAUDE.md`](CLAUDE.md). Design detail: [`docs/architecture.md`](docs/architecture.md). Open parameters: [`docs/decisions/`](docs/decisions/).

## Status

Implemented and tested: data loading + batch checks (`dataset.py`), background masking (background model + GrabCut; Otsu failed on real photos) + resize, GLCM / HSV-LAB / Canny feature extraction measured inside the copra only, whole-dataset feature table (8 rotations/flips per photo, cached for augmentation), outlier removal, StratifiedGroupKFold split, geometric augmentation, nested-CV Optuna tuning of the four models, model selection, evaluation metrics. Models are trained and compared (LightGBM selected); the model is not yet saved for deployment. Still `NotImplementedError`: explainability, artifact, Streamlit app.

Current export (`dataset/copra-dataset.csv`, dated 2026-10-14): 1000 photos in 250 batches — 200 photos (50 batches) each of classes `A`–`E`. Class F is still being collected. Class D was photographed under different lighting — a stated limitation (ADR-011).

## Run the pipeline

```powershell
.\run                       # menu: pick a process by number
.\run mask                  # background masking -> dataset/masked_data/ (for checking by eye)
.\run features              # -> data/processed/features.csv + mask_preview.jpg
.\run folds                 # -> features_clean.csv + folds.csv (outliers, folds)
.\run train                 # -> data/processed/results/ (nested CV, selection, metrics), ~80 min
.\run quick-train           # training smoke test, 3 trials per model
.\run all                   # features -> folds -> train, stops at the first failure
.\run train --n-trials 10   # extra arguments go to that process's script
```

`run.bat` calls `uv run python run.py`; the scripts it runs live in `scripts/`.

Check `mask_preview.jpg` by eye after every new export: red outline = detected copra.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 (`.python-version`) — newer versions break `llvmlite`'s build.

```bash
uv sync --extra dev
uv run pytest
```

## Project layout

```
src/copra_grading/
├── dataset.py         # Step 1: load CSV + photos, batch checks
├── preprocessing/     # Step 2: background masking (GrabCut; Otsu kept for comparison), resize
├── features/          # Step 3: GLCM texture, HSV/LAB color, Canny edges
├── cleaning.py        # Step 4a: drop outlier rows
├── splitting.py       # Step 4b: StratifiedGroupKFold (before augmentation)
├── augmentation.py    # Step 4c: rotate/flip, training folds only
├── models/            # Step 5: LR baseline, RF/XGBoost/LightGBM + Optuna
├── selection.py       # Step 6: best ensemble on Macro F1
├── evaluation.py      # Step 7: metrics
├── explainability.py  # Step 8: TreeSHAP by feature group
└── artifact.py        # Step 9: model + config serialization

app/streamlit_app.py   # Step 9: one photo in, one class out
configs/default.yaml   # all tunable parameters in one place
scripts/               # run the pipeline: mask_photos, extract_features, prepare_folds, train_models
tests/                 # mirrors src/ layout
docs/                  # architecture doc + ADRs
```

## Dataset layout

```
copra-dataset.csv      # id, batch_id, copra_class, path
photos/<id>.jpg
```

One row per photo. Each physical sample is photographed from 4 sides; those 4 rows share a `batch_id` and a `copra_class`. Same phone, 10 cm distance, same settings, mid-grey background.

## Pipeline

1. **Load** — CSV + photos, keep `batch_id` on every row.
2. **Preprocess** — background-model + GrabCut mask → apply mask → resize. (`process.md` names Otsu; it fails on the real photos, see ADR-008.)
3. **Features** — GLCM texture + HSV/LAB color + Canny edges, one row per photo.
4. **Prepare** — drop rows with any outlier value (|z| > 3 or outside 1.5×IQR, never imputed or relaxed); StratifiedGroupKFold(5), stratified by class and grouped by whole copra sample (several batches can share one sample); then rotate/flip augmentation on training folds only, more copies for smaller classes.
5. **Train** — Logistic Regression (standardized, baseline only), Random Forest, XGBoost, LightGBM; all class-weighted; Optuna on Macro F1.
6. **Select** — best of RF / XGBoost / LightGBM by Macro F1 across the 5 folds.
7. **Evaluate** — confusion matrix, accuracy, Macro F1, per-class precision/recall, per photo on held-out folds.
8. **Explain** — TreeSHAP summed by feature group (texture / color / edge).
9. **Deploy** — selected model in a Streamlit app: upload one photo, get a class A–F, optional per-photo SHAP.

```bash
uv run streamlit run app/streamlit_app.py   # once implemented
```
