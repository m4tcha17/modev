"""XGBoost ensemble candidate. Spec §8a. No feature scaling needed (tree-based)."""

from xgboost import XGBClassifier


def build_model(params: dict, sample_weight_strategy: str = "balanced") -> XGBClassifier:
    """Construct an XGBoost classifier from tuned params, class-weighted via sample_weight."""
    raise NotImplementedError
