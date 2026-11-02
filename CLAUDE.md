# Copra Moisture-Based Grade Classification — Project Rules

## Source of truth — follow strictly

Two files govern this repository. Follow them exactly; they override every other doc here (`process.md`, `goal.md`, `README.md`, `docs/architecture.md`, `docs/classes.md`, `docs/decisions/`) and any default behaviour.

1. **`docs/model-development-instructions.md`** — the build spec: *what* to build and *why*.
2. **`docs/model-acceptance-criteria.md`** — the pass/fail targets (P1–P4, M1–M4, L1–L2, F1–F2) and the results report format.

Rules for using them:

- Read both before any design call. If another doc disagrees with them, the two files win; update the other doc.
- If the two files disagree with **each other**, do not pick one. Flag it to the user (thesis team).
- Open items (instructions §15) are configurable defaults pending confirmation. Never present them as settled. ADRs in `docs/decisions/` record the defaults chosen.
- Targets are fixed. Never lower, round, or reinterpret one after seeing results. Never re-tune, retrain, or change features to push a metric over its target. Report the miss and the confused grades.

## Non-negotiable constraints (summary of the two files)

- **Classification only.** Six classes, `A`–`F`, read directly from the `copra_class` CSV column (the owner's pasa grade). Displayed as "Grade A" … "Grade F". Moisture bands: F < 6% ≤ A < 9% ≤ B < 12% ≤ C < 15% ≤ D < 18% ≤ E; a boundary value belongs to the higher grade (`docs/classes.md`). Never derive labels from a moisture reading. Never build a price/discount or regression head.
- **No other grade names** anywhere (no "Resecada", "Bodega", "Semi-Resecada", "Corriente", "Non-Merchantable", PCA "Class 1/2/3").
- **No end-to-end CNN / deep-learning image model.** Handcrafted features (GLCM + HSV/LAB + Canny) feeding tree ensembles is the deliberate architecture.
- **No meta-ensemble/stacking/voting** across Random Forest, XGBoost, LightGBM. Compared against each other and a Logistic Regression baseline, never blended.
- **No SMOTE or synthetic interpolation.** Class balance = native class weighting (all four models, LR included) + geometric augmentation.
- **No photometric augmentation** (brightness/contrast/color/hue), ever. Geometric only (rotation, flip), training portions only.
- **Outliers are dropped, never imputed, thresholds never relaxed.** |z| > 3 or outside the 1.5×IQR fence. A photo with even one outlier value is dropped as a whole row, so no null values remain (decided 2026-11-02, ADR-001). If too much data is lost, collect more data.
- **Preprocessing:** background model + GrabCut mask (Otsu failed), then lighting normalization to a fixed reference grey, then scale normalization. Features come only from the masked, normalized image.
- **No external datasets merged in**, no model-vs-meter agreement rate, no per-sample Brown-Duvel spot check.

## Hard ordering constraint — split before augment

Nested `StratifiedGroupKFold`, grouped by **`batch_id` (the Cube ID)**, stratified by `copra_class`:

- **Outer:** 5 folds, used only for reporting. Optuna never sees the held-out outer fold.
- **Inner:** 3 folds inside each outer training portion, for Optuna (mean inner Macro F1 per trial).
- Augment only inner training folds during tuning, and the whole outer training portion when refitting. Never augment before a split, at either level. LR standardization is fit on training data only (in a `Pipeline`).
- A cube's 4 face photos and all their augmented variants stay in one fold. No Cube ID may appear in both training and test of any fold (acceptance P4).
- There is no `copra_id`: cubes of the same copra can land in different folds. State this as a source of optimism in results.

## Selection and deployment

- Selected model = best of RF / XGBoost / LightGBM by mean outer-fold Macro F1. Logistic Regression is a baseline, never deployed.
- Only outer-fold, per-image scores count for acceptance. Inner-fold and training scores never do.
- Final artifact: re-run Optuna for the selected algorithm on the full dataset (grouped 5-fold), train on all data, serialize with its preprocessing/feature config. Streamlit app: one face photo in, one grade (A–F) out, optional per-photo TreeSHAP grouped by feature family.

## Git

**Never `git commit`.** Stage changes if useful, but leave committing to the user — commits in this repo are done manually, always.

## Other docs

`process.md` (short walkthrough), `docs/architecture.md` (module map + known gaps vs. the spec), `docs/classes.md`, `docs/decisions/` (ADRs for open parameters), `goal.md` (earlier targets T1–T8, superseded by the acceptance criteria). All are subordinate to the two source-of-truth files.
