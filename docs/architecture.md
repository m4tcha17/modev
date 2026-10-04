# Architecture — Copra Classification Pipeline

Source of truth: `../process.md`. This doc maps its 9 steps onto modules and describes data flow between them. Hard constraints live in `../CLAUDE.md`, not here.

## Pipeline stages

```mermaid
flowchart TD
    A[copra-dataset.csv + photos/\nid, batch_id, copra_class, path] --> B[Step 2: background-model + GrabCut mask, resize]
    B --> C[Step 3: GLCM + HSV/LAB + Canny features]
    C --> D[Step 4a: drop outlier rows\n|z| > 3 or outside 1.5×IQR]
    D --> E[Step 4b: StratifiedGroupKFold 5\ngrouped by whole sample, stratified by class]
    E --> F[Step 4c: geometric augmentation\ntraining folds only, class-weighted]
    F --> G[Step 5: train + Optuna on Macro F1\nLR, RandomForest, XGBoost, LightGBM]
    G --> H{Step 6: select best ensemble\nmean Macro F1 over 5 folds}
    H --> I[Step 7: evaluation]
    H --> J[Step 8: TreeSHAP by feature group]
    H --> K[Step 9: artifact → Streamlit app]
```

## Module → step map

| Module | Step | Responsibility |
|---|---|---|
| `dataset.py` | 1 | Class → moisture band map (`docs/classes.md`); load CSV, check columns and `copra_class` ∈ A–F, load RGB photos via `path`; report batches that don't have 4 photos, mix classes, or repeat an `id` |
| `preprocessing/background.py` | 2 | Default mask: fit the background's lighting gradient from the image border, flag pixels far from it, refine with GrabCut (ADR-008) |
| `preprocessing/otsu.py` | 2 | Otsu mask (comparison only — fails on real photos), apply mask (crop + zero background), resize |
| `preprocessing/pipeline.py` | 2 | `preprocess(image, config)`: pick mask method, apply, resize |
| `features/glcm.py` | 3a | Contrast, homogeneity, energy, entropy at 0/45/90/135° and configured distances |
| `features/color.py` | 3b | HSV + LAB per-channel mean/std over copra pixels only |
| `features/edges.py` | 3c | Canny edge density + contour stats |
| `features/combine.py` | 3 | Concatenate into one feature row per photo; GLCM/Canny measured inside the copra only (`features.exclude_boundary`) |
| `features/table.py` | 1–3 | Whole dataset → feature table (`id`, `batch_id`, `copra_class`, `mask_fraction`, features); run via `scripts/extract_features.py` |
| `cleaning.py` | 4a | Flag outlier values (z OR IQR), drop every row holding one |
| `splitting.py` | 4b | StratifiedGroupKFold(5), stratified by `copra_class`, grouped by `splitting.group_column` (whole sample; `batch_id` until a whole-sample ID exists), before augmentation |
| `augmentation.py` | 4c | Rotation/flip only, more copies for smaller classes (ADR-002) |
| `models/*.py` | 5 | LR (standardized, baseline) + RF/XGBoost/LightGBM (unscaled), all class-weighted |
| `models/tuning.py` | 5 | Optuna on Macro F1, same grouped + stratified folds (ADR-003) |
| `selection.py` | 6 | Per-fold Macro F1 for all four, pick best ensemble |
| `evaluation.py` | 7 | Confusion matrix, accuracy, Macro F1, per-class precision/recall |
| `explainability.py` | 8, 9 | TreeSHAP summed by group (texture/color/edge); per-photo explanation for the app |
| `artifact.py` | 9 | Bundle selected model + preprocessing/feature config (ADR-005) |
| `app/streamlit_app.py` | 9 | Upload one photo → class A–F, optional SHAP |

## Data flow

1. **Input**: one CSV row per photo; four rows share a `batch_id` and a `copra_class`. `path` resolves against the CSV's directory.
2. **Masking**: photo → background-model + GrabCut mask → cropped, background-zeroed image → fixed size.
3. **Features**: one row per photo, carrying `id`, `batch_id`, `copra_class` along.
4. **Outliers**: drop rows with any outlier value. Report how many rows went.
5. **Split → augment**: StratifiedGroupKFold first (grouped by whole sample, stratified by class); augmented copies of training-fold photos go through Steps 2–3.
6. **Train + tune**: all four models, class-weighted, Optuna on Macro F1.
7. **Select**: best ensemble by mean Macro F1 over 5 folds.
8. **Evaluate** on held-out folds, per photo.
9. **Explain** with TreeSHAP; **deploy** the selected model + config.

## Placeholder data

`scripts/make_synthetic_dataset.py` (stub) generates a schema-matching dataset (4 photos per `batch_id`, classes A–F) for testing the wiring when real data is thin. The first real export (`dataset/`, 2026-10-04) has 34 batches, all class `A`.
