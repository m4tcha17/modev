# Copra Moisture-Based Grade Classification — Project Rules

Full build spec: `model-development-instructions.md`. Architecture detail: `docs/architecture.md`. Unresolved parameters: `docs/decisions/` (ADR-001..007). This file holds only the hard, stable constraints — read the other two for everything else.

## Non-negotiable constraints

- **Classification only.** Never build a price/discount regression head or output. The three classes are `1`, `2`, `3` (or `class_1`/`class_2`/`class_3`) — never a commercial grade name (no "Resecada", "Bodega", "Corriente", etc.) anywhere in code, labels, configs, or generated output.
- **No end-to-end CNN / deep-learning image model.** Handcrafted features (GLCM + HSV/LAB + Canny) feeding tree ensembles is the deliberate architecture, not a placeholder for one.
- **No meta-ensemble/stacking** across Random Forest, XGBoost, LightGBM. They are compared against each other and against a Logistic Regression baseline — never blended into one model.
- **No SMOTE or synthetic interpolation** for class balance. Class balance correction is class weighting (model-level, all four models) + real geometric augmentation (data-level) only.
- **No photometric augmentation** (brightness/contrast/color/hue shifts), ever — it corrupts the HSV/LAB color signal the pipeline measures. Augmentation is geometric only (rotation, flip).
- **No external "wet/dried" dataset merged in** without an explicit, separately-agreed labeling protocol. Default assumption: field-collected data only.

## Hard ordering constraint

**Split before augment, never the reverse.** GroupKFold (grouped by `Sample_ID`) must run before any augmented copies are generated, so every augmented variant of a sample's images stays in the same fold as the original. Splitting after augmentation, or without grouping by `Sample_ID`, leaks near-duplicate samples across folds and silently inflates validation/test scores.

## Algorithm selection vs. deployment configuration — never conflate

Two distinct stages, kept separate in code:

1. **Algorithm selection**: all four models (LR baseline + 3 ensembles) trained/tuned on the **combined all-angle** feature set. Best-scoring ensemble (by tuned Macro F1) is the selected algorithm. LR is baseline reference only, never deployable.
2. **Deployment configuration**: only the *selected* algorithm is retrained on single-angle / angle-subset feature sets to find the minimal photo input a live user submits.

The combined-all-angle model from step 1 is never the deployed artifact — it exists only to pick the algorithm. Step 2's retrain is mandatory before anything is called "the deployed model."

## Everything else

Feature families, tuning targets, evaluation requirements, serialization format, and the 7 open/unconfirmed parameters live in `docs/architecture.md` and `docs/decisions/`. Consult those, and `model-development-instructions.md` itself, before making a design call not covered above.
