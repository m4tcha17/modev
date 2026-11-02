"""LightGBM ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

from lightgbm import LGBMClassifier


def build_model(params: dict, n_jobs: int = 1, random_state: int = 42) -> LGBMClassifier:
    """Construct a class_weight='balanced' LightGBM classifier from tuned params.

    subsample_freq=1 so a tuned `subsample` < 1 actually takes effect
    (LightGBM ignores bagging when the frequency is 0).
    """
    return LGBMClassifier(
        **params,
        class_weight="balanced",
        subsample_freq=1,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
