# ADR-003: Optuna Search Space and Trial Budget

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** §8b, §15 item 3

## Default

Trial budget: 50-100 trials per model, given a post-augmentation dataset in the low hundreds to low thousands of rows — enough to explore the space without excessive compute for a two-person team.

Search spaces (starting points, `models/tuning.py`):

| Model | Parameters |
|---|---|
| Logistic Regression | `C` (log-uniform, 1e-3 to 1e2), `penalty` (`l1`/`l2`) |
| Random Forest | `n_estimators` (100-500), `max_depth` (3-20), `min_samples_leaf` (1-10), `max_features` (`sqrt`/`log2`/None) |
| XGBoost | `n_estimators` (100-500), `max_depth` (3-10), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_weight` (1-10), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |
| LightGBM | `n_estimators` (100-500), `num_leaves` (7-127), `learning_rate` (log-uniform, 1e-3 to 3e-1), `min_child_samples` (5-50), `subsample`/`colsample_bytree` (0.5-1.0), `reg_alpha`/`reg_lambda` (log-uniform, 1e-3 to 10) |

Objective: Macro F1 (see `../../CLAUDE.md` and instructions §8b — never raw accuracy).

## Rationale

Small dataset means small search spaces and modest trial budgets are appropriate; overly wide spaces or thousands of trials risk overfitting the *hyperparameter search itself* to a small validation fold.

## What would change it

If cross-validated variance across Optuna trials looks unstable, widen the budget; if a model's tuned params consistently land at a search-space boundary, widen that specific range.

## Owner

Thesis team (methodology decision), can be revisited by the implementing session based on observed tuning behavior.
