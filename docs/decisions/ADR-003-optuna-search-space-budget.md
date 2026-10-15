# ADR-003: Optuna Search Space and Trial Budget

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** `process.md` Step 5

## Default

Trial budget: 50-100 trials per model, given a post-augmentation dataset in the low hundreds to low thousands of rows — enough to explore the space without excessive compute for a two-person team.

Search spaces (starting points, `models/tuning.py`):

| Model | Parameters |
|---|---|
| Logistic Regression | `C` (log-uniform, 1e-3 to 1e2), `penalty` (`l1`/`l2`) |
| Random Forest | `n_estimators` (100-500), `max_depth` (3-20), `min_samples_leaf` (1-10), `max_features` (`sqrt`/`log2`/None) |
| XGBoost | `n_estimators` (100-500), `max_depth` (3-10), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_weight` (1-10), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |
| LightGBM | `n_estimators` (100-500), `num_leaves` (7-127), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_samples` (5-50), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |

Objective: Macro F1, never raw accuracy. `process.md` requires regularization tuning: `max_depth` + `min_samples_leaf` (RF), `max_depth` + `min_child_weight` (XGBoost, LightGBM) — LightGBM's `min_child_weight` should join or replace `min_child_samples` above, and LightGBM needs an explicit `max_depth` range (e.g. 3-12) alongside `num_leaves`.

Tuning CV must group by `batch_id`, same as the evaluation split.

### Nested CV (decided 2026-10-14)

Tuning runs inside each of the 5 outer folds: the outer training photos are re-split with `folds.make_folds` (StratifiedGroupKFold on `batch_id`, split before augment) into `tuning.inner_n_splits` = 3 inner folds, and Optuna maximises mean inner Macro F1. The outer validation fold is never seen during tuning, so the reported Macro F1 is not inflated by hyperparameter search. 3 inner folds instead of 5 keeps the cost near 1-1.5 h for all four models at 75 trials (measured: worst-case single fit about 2 s on 12 cores).

Changes to the table above:
- Logistic Regression: scikit-learn 1.9 deprecates `penalty`; `l1`/`l2` is expressed as `l1_ratio` 1.0/0.0 with `solver="saga"`.
- LightGBM: adds `max_depth` (3-12) and `min_child_weight` (log-uniform, 1e-3 to 10), as required by `process.md`; `min_child_samples` stays.

### Narrowed tree spaces (tried 2026-10-15, rejected)

First full nested-CV run (5-class export, 65 features): Logistic Regression 0.777 mean Macro F1, beating LightGBM 0.734, XGBoost 0.722 and Random Forest 0.676, winning or tying on every fold. The tuned trees went deep (RF `max_depth` 10-16, XGBoost up to 10) on ~190 training samples per outer fold. That memorizes batches, and the rotated copies add near-duplicate rows. The spaces were narrowed toward shallow, regularized trees:

| Model | Parameter | Was | Now |
|---|---|---|---|
| Random Forest | `max_depth` | 3-20 | 2-8 |
| Random Forest | `min_samples_leaf` | 1-10 | 5-40 |
| XGBoost | `max_depth` | 3-10 | 2-6 |
| XGBoost | `min_child_weight` | 1-10 | 1-20 |
| XGBoost | `reg_lambda` | 1e-3-10 (log) | 1e-1-50 (log) |
| LightGBM | `num_leaves` | 7-127 | 4-31 |
| LightGBM | `max_depth` | 3-12 | 2-6 |
| LightGBM | `min_child_samples` | 5-50 | 10-60 |
| LightGBM | `reg_lambda` | 1e-3-10 (log) | 1e-1-50 (log) |

Leaf sizes count training rows, and those include augmented copies (up to 4 per photo), so a leaf of 5 rows can still be a single photo. Logistic Regression is unchanged. Applied together with feature pruning (ADR-012).

**Result (full nested CV, 75 trials, together with ADR-012 pruning at 0.95):**

| Model | Mean Macro F1, original spaces | Narrowed + pruned |
|---|---|---|
| Logistic Regression | 0.777 | 0.757 |
| LightGBM | 0.734 | 0.713 |
| XGBoost | 0.722 | 0.704 |
| Random Forest | 0.676 | 0.680 |

No gain, and the gap between Logistic Regression and the best ensemble stayed at ~0.04. Random Forest picked `max_depth` 8 (the new maximum) in 4 of 5 folds, so the deep trees of the first run were not plain overfitting. **Reverted to the original spaces above.** Conclusion: at ~235 samples the linear baseline genuinely fits these features better. The thesis team decided not to try further fixes; report this as a finding.

## Rationale

Small dataset means small search spaces and modest trial budgets are appropriate; overly wide spaces or thousands of trials risk overfitting the *hyperparameter search itself* to a small validation fold.

## What would change it

If cross-validated variance across Optuna trials looks unstable, widen the budget; if a model's tuned params consistently land at a search-space boundary, widen that specific range.

## Owner

Thesis team (methodology decision), can be revisited by the implementing session based on observed tuning behavior.
