"""Random Forest ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

from sklearn.ensemble import RandomForestClassifier


def build_model(params: dict, n_jobs: int = 1, random_state: int = 42) -> RandomForestClassifier:
    """Construct a class_weight='balanced' Random Forest from tuned params."""
    return RandomForestClassifier(
        **params, class_weight="balanced", n_jobs=n_jobs, random_state=random_state
    )
