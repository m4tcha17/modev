# Architecture — Copra Grade Classification Pipeline

Source of truth: `model-development-instructions.md` (build spec, cited as §N) and `model-acceptance-criteria.md` (targets P/M/L/F). This doc maps the spec onto modules and lists where the code does not yet match it. Hard constraints live in `../CLAUDE.md`.

## Pipeline stages

```mermaid
flowchart TD
    A[copra-dataset.csv + photos/\nid, batch_id = Cube ID, copra_class, path] --> B[§4: background model + GrabCut mask\nlighting normalization, resize]
    B --> C[§5: GLCM + HSV/LAB + Canny features]
    C --> D[§6: drop outlier rows\n|z| > 3 or outside 1.5×IQR]
    D --> E[§7a/§8b: outer StratifiedGroupKFold 5\ngrouped by batch_id, stratified by class]
    E --> F[inner StratifiedGroupKFold 3\nOptuna on Macro F1, augment inner train only]
    F --> G[refit on augmented outer train\nscore held-out outer fold]
    G --> H{§8c: select best of RF / XGB / LGBM\nmean outer-fold Macro F1}
    H --> I[§11: evaluation + acceptance table]
    H --> J[§10: TreeSHAP by feature family]
    H --> K[§12: retune on full data → artifact → Streamlit]
```

## Module → spec map

| Module | Spec | Responsibility |
|---|---|---|
| `dataset.py` | §3, §6.1 | Grade → moisture band map (`docs/classes.md`); load CSV, check columns and `copra_class` ∈ A–F, load photos via `path`; report cubes without 4 photos, mixed grades, repeated `id` |
| `preprocessing/background.py` | §4 | Background model + GrabCut mask (ADR-008) |
| `preprocessing/otsu.py` | §4 | Otsu mask, comparison only (failed on the pilot); apply mask, resize |
| `preprocessing/lighting.py` | §4 | Scale paper to `reference_grey` after masking (ADR-012) |
| `preprocessing/pipeline.py` | §4 | `preprocess(image, config)`: mask → lighting → resize; mask method behind one function |
| `features/glcm.py` | §5a | Contrast, homogeneity, energy, entropy at 0/45/90/135° and configured distances (ADR-006) |
| `features/color.py` | §5b | HSV + LAB per-channel mean/SD over copra pixels; circular hue stats (ADR-009) |
| `features/edges.py` | §5c | Canny edge density + contour stats |
| `features/combine.py` | §5 | One feature row per face image; GLCM/Canny measured inside the copra only |
| `features/table.py` | §3–§5 | Dataset → feature table (`id`, `batch_id`, `copra_class`, `mask_fraction`, features); `scripts/extract_features.py` |
| `cleaning.py` | §6.2 | Flag outlier values (z OR IQR), drop every row holding one (ADR-001) |
| `splitting.py` | §7a | StratifiedGroupKFold by `splitting.group_column` = `batch_id` (ADR-010), before augmentation |
| `augmentation.py` | §7b | Rotation/flip only, more copies for smaller grades (ADR-002) |
| `folds.py` | §7 | Split originals, then pick augmented training copies per fold; validation = originals only. `scripts/prepare_folds.py` |
| `models/*.py` | §8a | LR (standardized, baseline) + RF/XGBoost/LightGBM (unscaled), all class-weighted |
| `models/tuning.py` | §8b | Optuna on Macro F1 (ADR-003) |
| `selection.py` | §8c | Per-fold Macro F1 for all four, pick best ensemble |
| `evaluation.py` | §11 | Confusion matrix, accuracy, Macro F1, per-grade precision/recall |
| `background_check.py` | §11, L2 | Paper-only classifier before/after normalization; `scripts/background_check.py` |
| `explainability.py` | §10 | TreeSHAP by family; per-photo explanation for the app |
| `artifact.py` | §12 | Selected model + preprocessing/feature config (ADR-005) |
| `app/streamlit_app.py` | §12 | One photo in → grade A–F, optional SHAP (separate, later effort) |

## Known gaps vs. the spec (as of 2026-11-02)

Code that does not yet do what the two source-of-truth files require. Fix before reporting results.

| Gap | Spec | Current state |
|---|---|---|
| Tuning not nested | §8b, acceptance §1.3 | `scripts/train_models.py` tunes and scores on the same 5 folds; needs 3 inner folds per outer fold, refit, score outer |
| Final full-data retune + train | §8b, §12 | Not built |
| Acceptance thresholds as config + pass/fail table | acceptance §3–§6 | Only `targets.background_check_margin` exists; P1, P2, M1–M4, F1 thresholds and the §6 table are missing |
| Background check pass mark | acceptance L2 | Code uses chance + margin; spec pass mark is accuracy ≤ 25% (six grades) |
| Outlier counts per grade | acceptance P3, §6.2 | Total drops reported, not per grade |
| Duplicate Cube ID resolution across merged exports | §6.1 | Repeated `id` reported; duplicate Cube ID resolution not built |
| Feature-family ablation | §11, F1 | Not built |
| Brown-Duvel label check | §9, L1 | Not built (needs a small `class, reading` file) |
| TreeSHAP, artifact, Streamlit, synthetic data | §10, §12, §15 | `NotImplementedError` stubs |
| PCA / t-SNE sanity plot | §11 | Not built (optional) |

## Data flow

1. **Input:** one CSV row per face photo; 4 rows share a `batch_id` (Cube ID) and a `copra_class`.
2. **Preprocess:** mask → lighting normalization → fixed size.
3. **Features:** one row per photo, carrying `id`, `batch_id`, `copra_class`.
4. **Outliers:** drop rows with any outlier value; report drops per grade.
5. **Nested CV:** outer split → inner split → augment inner training → Optuna → refit on augmented outer training → score outer fold.
6. **Select:** best ensemble by mean outer-fold Macro F1.
7. **Evaluate** per image on outer folds; fill in the acceptance table.
8. **Explain** with TreeSHAP; **retune and train on all data**, serialize the artifact.

## Placeholder data

`scripts/make_synthetic_dataset.py` (stub) is meant to generate a schema-matching dataset (4 photos per `batch_id`, grades A–F) to prove the code path when real data is thin (§15).
