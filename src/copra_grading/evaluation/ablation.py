"""Feature-family ablation: color-only / texture-only / edge-only / combined. Spec §11.

Justifies keeping all three feature families in the default pipeline (§5),
particularly Canny edge density, the newest addition. Standard evaluation
step, not an afterthought.
"""

import pandas as pd


def run_ablation(
    X_by_family: dict[str, pd.DataFrame], y: pd.Series, groups: pd.Series
) -> dict[str, float]:
    """Train the selected algorithm on each family subset and combined; return Macro F1 per config."""
    raise NotImplementedError
