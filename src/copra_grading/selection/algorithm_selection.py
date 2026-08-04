"""Algorithm selection: combined all-angle training of all four models. Spec §8c step 1.

Whichever ensemble (Random Forest / XGBoost / LightGBM) scores best on tuned
Macro F1 is the selected algorithm. Logistic Regression is baseline reference
only, never eligible for deployment regardless of its score. The combined
all-angle model produced here is NOT the deployment artifact - it exists only
to answer "which algorithm is strongest." See deployment_config.py for the
mandatory next stage.
"""

import pandas as pd


def select_algorithm(
    X_combined: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> str:
    """Train/tune all four models on the combined all-angle feature set,
    return the name of the best-scoring ensemble ("random_forest" | "xgboost" | "lightgbm").
    """
    raise NotImplementedError
