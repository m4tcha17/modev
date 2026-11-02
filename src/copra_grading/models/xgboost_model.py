"""XGBoost ensemble candidate. process.md Step 5. No feature scaling needed (tree-based).

XGBClassifier has no class_weight argument: class weighting is applied as
balanced per-row sample weights at fit time (models.fit_predict).
"""

from xgboost import XGBClassifier


def build_model(params: dict, n_jobs: int = 1, random_state: int = 42) -> XGBClassifier:
    """Construct an XGBoost classifier from tuned params, class-weighted via sample_weight."""
    return XGBClassifier(
        **params,
        objective="multi:softprob",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=0,
    )
