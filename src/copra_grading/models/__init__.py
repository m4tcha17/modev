"""Four candidate models + Optuna tuning. Spec §8a, §8b.

All four are natively class-weighted, including the Logistic Regression
baseline (deliberate - keeps comparison isolating algorithm strength, not
which models got imbalance help). Optuna optimizes Macro F1, never raw
accuracy. Search spaces/budgets: see ADR-003.
"""
