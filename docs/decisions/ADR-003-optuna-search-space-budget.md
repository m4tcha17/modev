# ADR-003: Optuna Search Space and Trial Budget

**Status:** proposed-default, pending thesis-team confirmation (open item §15.2). Nested structure is fixed by the spec (decided 2026-10-14).
**Spec reference:** `model-development-instructions.md` §8b; acceptance §1.3

## Fixed by the spec (not open)

- Objective: **Macro F1**, never raw accuracy.
- **Nested CV.** Outer: the 5 `StratifiedGroupKFold` folds (grouped by `batch_id`, stratified by `copra_class`), reporting only; Optuna never sees the held-out outer fold. Inner: `StratifiedGroupKFold(n_splits=3)` inside each outer training portion, same grouping; each trial scored by mean inner Macro F1.
- Augmentation only on inner training folds during tuning, and on the whole outer training portion when refitting. LR standardization inside a `Pipeline`, fit on training data only.
- Per outer fold: refit the best trial on the augmented outer training portion, score once on the outer fold. Report each outer fold's chosen hyperparameters.
- Final model: rerun Optuna for the selected algorithm on the full dataset (grouped 5-fold), then train on all data.

## Default (open part)

Trial budget: **75 trials per model** (`tuning.n_trials`), per outer fold under nested CV. The dataset is a few thousand rows after augmentation, so a modest budget is enough.

Search spaces (starting points, `models/tuning.py`):

| Model | Parameters |
|---|---|
| Logistic Regression | `C` (log-uniform, 1e-3 to 1e2), L1/L2 via `l1_ratio` |
| Random Forest | `n_estimators` (100-500), `max_depth` (3-20), `min_samples_leaf` (1-10), `max_features` (`sqrt`/`log2`/None) |
| XGBoost | `n_estimators` (100-500), `max_depth` (3-10), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_weight` (1-10), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |
| LightGBM | `n_estimators` (100-500), `num_leaves` (7-127), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_samples` (5-50), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |

These are this repo's choices, not decisions the thesis team made. Check `models/tuning.py` for the exact current ranges.

## Current gap

`scripts/train_models.py` tunes and scores on the same 5 folds (not nested). Results from it are not valid for acceptance until nested tuning is built.

## Rationale

Small dataset means small search spaces and modest budgets; wide spaces or thousands of trials risk overfitting the search to small inner folds.

## What would change it

Unstable chosen hyperparameters across outer folds: widen the budget. Tuned params repeatedly at a range boundary: widen that range. Never change the search to push a metric over an acceptance target.

## Owner

Thesis team; the implementing session may revisit based on observed tuning behaviour.
