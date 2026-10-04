"""Random Forest ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

from sklearn.ensemble import RandomForestClassifier


def build_model(params: dict) -> RandomForestClassifier:
    """Construct a class_weight='balanced' Random Forest from tuned params."""
    raise NotImplementedError
