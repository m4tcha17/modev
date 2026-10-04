# Copra Classification — Project Rules

Build spec: `process.md`. Architecture detail: `docs/architecture.md`. Open parameters: `docs/decisions/`. This file holds only the hard, stable constraints — read the other two for everything else.

## Non-negotiable constraints

- **Classification only.** Six classes, `A`–`F`, read directly from the `copra_class` CSV column. Each stands for a moisture band (F < 6% < A < 9 < B < 12 < C < 15 < D < 18% ≤ E; see `docs/classes.md`) — letters are not in moisture order. Never derive labels from anything else, never build a price/discount regression head or output.
- **No end-to-end CNN / deep-learning image model.** Handcrafted features (GLCM + HSV/LAB + Canny) feeding tree ensembles is the deliberate architecture, not a placeholder for one.
- **No meta-ensemble/stacking** across Random Forest, XGBoost, LightGBM. They are compared against each other and against a Logistic Regression baseline — never blended into one model.
- **No SMOTE or synthetic interpolation** for class balance. Class balance correction is class weighting (model-level, all four models) + geometric augmentation (data-level) only.
- **No photometric augmentation** (brightness/contrast/color/hue shifts), ever — it corrupts the HSV/LAB color features. Augmentation is geometric only (rotation, flip), training folds only.
- **Outliers are dropped, never imputed, thresholds never relaxed.** A feature value is an outlier if |z| > 3 or outside the 1.5×IQR fence; any photo row with one is dropped. If too much data is lost, collect more data.

## Hard ordering constraint

**Split before augment, never the reverse.** StratifiedGroupKFold (5 folds, stratified by `copra_class`) must run before any augmented copies are generated. Group by the **whole copra sample**: every photo of it — across all its batches — and every augmented variant stays in the same fold. Several `batch_id`s can come from one whole sample, so grouping by `batch_id` alone can leak; set `splitting.group_column` to a whole-sample ID once the CSV has one. Splitting after augmentation, or with the wrong grouping, puts part of a sample in training and part in testing and silently inflates scores.

## Selection and deployment

All four models are compared on Macro F1 across the 5 folds. The best of Random Forest / XGBoost / LightGBM is the selected model and is what gets deployed. Logistic Regression is a reference point only, never deployed. Deployment is a Streamlit app: one photo in, one class (A–F) out, optional per-photo SHAP grouped by feature type.

## Git

**Never `git commit`.** Stage changes if useful, but leave committing to the user — commits in this repo are done manually, always.

## Everything else

Feature families, tuning targets, evaluation metrics, serialization format, and open parameters live in `docs/architecture.md` and `docs/decisions/`. Consult those, and `process.md` itself, before making a design call not covered above.
