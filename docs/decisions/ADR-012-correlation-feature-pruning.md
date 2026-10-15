# ADR-012: Correlation-Based Feature Pruning

**Status:** rejected 2026-10-15. Code removed in the cleanup that followed; this ADR is the record of the experiment.
**Spec reference:** `process.md` Step 5 (not named in the spec; added after the first full run)

## Decision

As tested, each outer training fold dropped redundant features before tuning. A greedy pass in extraction order kept a feature only if its |Spearman rho| with every already-kept feature was at most 0.95.

- **Fitted per outer fold, on that fold's training photos only.** The validation fold never influences the list. Inner tuning folds reuse the outer fold's list.
- **Unsupervised.** Labels are never used, so no class information leaks into selection.
- **Original photos only.** Augmented copies are near-duplicates and would overweight photos with more copies.
- **Same list for all four models**, Logistic Regression included, so the comparison still isolates algorithm strength.
- `scripts/train_models.py` wrote the per-fold lists to `results/selected_features.json`.

## Rationale

In the first full nested-CV run, Logistic Regression beat all three tree ensembles (ADR-003, 2026-10-15). 65 features for ~235 samples is a lot, and many are near-copies: one GLCM property at three distances and four angles, or paired HSV/LAB channels. Features kept on the 2026-10-14 export, original photos only:

| threshold | features kept |
|---|---|
| 0.99 | 43 |
| 0.95 | 27 |
| 0.90 | 20 |
| 0.85 | 18 |

Spearman is used instead of Pearson because monotone but non-linear relations (e.g. GLCM energy vs homogeneity) are redundant for trees too.

Not done:
- Supervised selection (mutual information, model importance). It would need its own nested loop to avoid leakage.
- Dimensionality reduction (PCA). It destroys the feature-type grouping that SHAP needs for the app.

## Result

Full nested CV at 0.95 (27-28 features per fold), run together with the narrowed tree spaces (ADR-003): mean Macro F1 fell for Logistic Regression (0.777 to 0.757), LightGBM (0.734 to 0.713) and XGBoost (0.722 to 0.704); Random Forest was flat (0.676 to 0.680). Logistic Regression lost as much as the trees, so the pruned features were not pure duplicates: pairs above |rho| 0.95 still carried signal. Pruning was removed. The thesis team decided not to tune the threshold further.

## Consequences

None: the pipeline uses all features. If ever revived (recoverable from git history, commit 524ee9e), the deployed model would need the pruned list from its final fit stored with it.

## What would change it

Only a thesis-team decision to revisit it, e.g. after a much larger dataset (more classes or samples) changes the feature-to-sample ratio.

## Owner

Thesis team (methodology decision).
