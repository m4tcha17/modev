"""Four candidate models + Optuna tuning. process.md Step 5.

All four are natively class-weighted, including the Logistic Regression
baseline (deliberate - keeps comparison isolating algorithm strength, not
which models got imbalance help). Optuna optimizes Macro F1, never raw
accuracy. Regularization tuned: max depth + min samples per leaf (RF),
max depth + min child weight (XGBoost, LightGBM). Search spaces/budgets:
see ADR-003.
"""
