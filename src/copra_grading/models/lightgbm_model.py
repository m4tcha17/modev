"""LightGBM ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

from lightgbm import LGBMClassifier


def build_model(params: dict) -> LGBMClassifier:
    """Construct a class_weight='balanced' LightGBM classifier from tuned params."""
    raise NotImplementedError
