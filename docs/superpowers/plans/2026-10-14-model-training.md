# Model Training, Selection and Evaluation (Steps 5-7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the four classifiers (Logistic Regression baseline, Random Forest, XGBoost, LightGBM) with nested cross-validation, select the best tree ensemble on Macro F1, and report per-photo metrics on the held-out folds.

**Architecture:** Each model module owns its estimator and its Optuna search space. A small registry in `models/__init__.py` gives one way to build, weight, fit and predict with string labels. Nested CV: the 5 outer folds from `folds.make_folds` are never seen by Optuna; inside each outer training fold, `tuning.tune_model` re-splits that fold's photos with `make_folds` (split before augment, grouped by `batch_id`) into inner folds and maximises mean inner Macro F1. `selection.cross_validate_models` runs the outer loop and collects per-fold scores, per-fold best params and out-of-fold predictions; `evaluation.compute_metrics` turns those predictions into the Step 7 report. `scripts/train_models.py` runs it all on `data/processed/features_clean.csv`.

**Tech Stack:** Python 3.12, pandas, scikit-learn 1.9, xgboost 3.3, lightgbm 4.7, optuna 4.9, pytest, `uv run`.

**Spec:** `process.md` Steps 5-7, `CLAUDE.md`, `docs/architecture.md`, `docs/decisions/ADR-003-optuna-search-space-budget.md`.

## Global Constraints

- Labels come only from the `copra_class` column. Six classes `A`-`F`; class F is not in the data yet, so every component must work with any subset of the six.
- No meta-ensemble, stacking or blending. The four models are trained and scored separately.
- No SMOTE or synthetic interpolation. Class balance = class weighting on all four models + geometric augmentation already chosen by `folds.make_folds`.
- No photometric augmentation. This plan adds no augmentation at all; it only reuses the rows `make_folds` picks.
- Split before augment, grouped by `splitting.group_column` (`batch_id`). Inner tuning folds obey the same rule.
- Validation and scoring are per photo, on original (`augment == "rot0"`) rows only.
- Optuna objective is Macro F1, never accuracy. `tuning.n_trials: 75` per model per outer fold (ADR-003).
- Regularization tuned: `max_depth` + `min_samples_leaf` (RF), `max_depth` + `min_child_weight` (XGBoost, LightGBM).
- Logistic Regression is the only scaled model and is never selectable or deployable.
- Selection: best mean Macro F1 over the 5 outer folds among `random_forest`, `xgboost`, `lightgbm`.
- **Never `git commit`.** Stage with `git add`; the user commits manually.
- Work on branch `model-training`, not `main`.

## Review Focus

1. **A class missing from a training split.** Class F is absent everywhere; a small class can also be thin in an inner fold. Expected: fitting works with whatever classes are present, and metrics still list a class that occurs in `y_true` but is never predicted (recall 0). Pinned in Task 1 (`test_fit_with_five_classes_predicts_string_labels`) and Task 4 (`test_class_never_predicted_has_zero_recall`).
2. **Labels leaking out as integers.** XGBoost needs integer labels; a person reading predictions expects `"A"`-`"F"`. Pinned in Task 1 (`test_predictions_are_class_letters_for_every_model`).
3. **Outer test photos influencing tuning.** Expected: Optuna only sees the outer training photos. Pinned in Task 3 (`test_tuning_never_sees_outer_validation_photos`).
4. **Same seed, different result.** Expected: rerunning with the same config gives the same best params and scores. Pinned in Task 2 (`test_same_seed_same_best_params`).
5. **Logistic Regression scoring highest.** Expected: it is reported but `select_model` still returns an ensemble. Pinned in Task 3 (`test_select_model_ignores_logistic_regression`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/copra_grading/models/logreg.py` | LR pipeline (StandardScaler + LR) and its search space |
| `src/copra_grading/models/random_forest.py` | RF estimator and its search space |
| `src/copra_grading/models/xgboost_model.py` | XGBoost estimator and its search space |
| `src/copra_grading/models/lightgbm_model.py` | LightGBM estimator and its search space |
| `src/copra_grading/models/__init__.py` | Registry: `MODEL_NAMES`, `suggest_params`, `fit_model`, `FittedModel` |
| `src/copra_grading/models/tuning.py` | Inner-CV Optuna study for one model on one outer training set |
| `src/copra_grading/selection.py` | Outer nested-CV loop, `select_model` |
| `src/copra_grading/evaluation.py` | Step 7 metrics from out-of-fold predictions |
| `scripts/train_models.py` | CLI: run everything, write results to `data/processed/results/` |
| `configs/default.yaml` | Add `tuning.inner_n_splits`, `tuning.random_state` |
| `docs/decisions/ADR-003-optuna-search-space-budget.md` | Record nested CV, inner split count, LightGBM/LR space changes |
| `tests/models/test_models.py`, `tests/models/test_tuning.py`, `tests/test_selection.py`, `tests/test_evaluation.py` | Tests |

Shared test fixture (used by Tasks 2 and 3), add to `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest

from copra_grading.augmentation import TRANSFORMS

CONFIG = {
    "splitting": {"n_splits": 5, "group_column": "batch_id",
                  "label_column": "copra_class", "random_state": 42},
    "augmentation": {"base_copies": 1, "max_multiplier": 4},
    "tuning": {"n_trials": 2, "inner_n_splits": 3, "random_state": 42,
               "objective_metric": "f1_macro"},
}


@pytest.fixture
def config():
    return {k: dict(v) for k, v in CONFIG.items()}


@pytest.fixture
def feature_table():
    """Feature table shaped like features_clean.csv: 40 batches x 4 photos x
    8 transforms, classes A-E (no F), two informative features."""
    rng = np.random.default_rng(0)
    classes = ("A", "B", "C", "D", "E")
    rows = []
    for b in range(40):
        cls = classes[b % 5]
        centre = classes.index(cls) * 3.0
        for p in range(4):
            base = centre + rng.normal(0, 0.3, size=2)
            for t in TRANSFORMS:
                f1, f2 = base + rng.normal(0, 0.05, size=2)
                rows.append({"id": f"B{b}-{p}", "batch_id": f"B{b}", "copra_class": cls,
                             "augment": t, "mask_fraction": 0.3,
                             "feat_1": f1, "feat_2": f2})
    return pd.DataFrame(rows)
```

---

### Task 0: Branch, config and ADR-003

**Files:**
- Modify: `configs/default.yaml` (the `tuning:` block)
- Modify: `docs/decisions/ADR-003-optuna-search-space-budget.md`
- Create: `tests/conftest.py` (fixture above)

- [ ] **Step 1: Create the branch**

Run: `git switch -c model-training`
Expected: `Switched to a new branch 'model-training'`

- [ ] **Step 2: Extend the tuning config**

Replace the `tuning:` block in `configs/default.yaml` with:

```yaml
tuning:
  n_trials: 75 # per model, per outer fold (ADR-003)
  objective_metric: "f1_macro"
  # Nested CV: Optuna tunes inside each outer training fold on its own
  # grouped inner folds, so the 5 outer folds stay unseen (ADR-003).
  inner_n_splits: 3
  random_state: 42
```

- [ ] **Step 3: Update ADR-003**

Append under `## Default`:

```markdown
### Nested CV (decided 2026-10-14)

Tuning runs inside each of the 5 outer folds: the outer training photos are re-split with `folds.make_folds` (StratifiedGroupKFold on `batch_id`, split before augment) into `tuning.inner_n_splits` = 3 inner folds, and Optuna maximises mean inner Macro F1. The outer validation fold is never seen during tuning, so the reported Macro F1 is not inflated by hyperparameter search. 3 inner folds instead of 5 keeps the cost near 1-1.5 h for all four models at 75 trials (measured: worst-case single fit about 2 s on 12 cores).

Changes to the table above:
- Logistic Regression: scikit-learn 1.9 deprecates `penalty`; `l1`/`l2` is expressed as `l1_ratio` 1.0/0.0 with `solver="saga"`.
- LightGBM: adds `max_depth` (3-12) and `min_child_weight` (log-uniform, 1e-3 to 10), as required by `process.md`; `min_child_samples` stays.
```

- [ ] **Step 4: Write the shared fixture**

Create `tests/conftest.py` with the fixture code from the File Structure section.

- [ ] **Step 5: Run the existing suite**

Run: `uv run pytest -q`
Expected: `95 passed`

- [ ] **Step 6: Stage**

```bash
git add configs/default.yaml docs/decisions/ADR-003-optuna-search-space-budget.md tests/conftest.py
```

---

### Task 1: The four models and the registry

**Files:**
- Modify: `src/copra_grading/models/logreg.py`, `random_forest.py`, `xgboost_model.py`, `lightgbm_model.py`, `__init__.py`
- Test: `tests/models/test_models.py`

**Interfaces:**
- Produces:
  - each model module: `build_model(params: dict, seed: int) -> estimator`, `suggest_params(trial: optuna.Trial) -> dict`
  - `models.MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost", "lightgbm")`
  - `models.suggest_params(name: str, trial: optuna.Trial) -> dict`
  - `models.fit_model(name: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel`
  - `FittedModel.predict(X: pd.DataFrame) -> np.ndarray` of class letters; `FittedModel.estimator`; `FittedModel.classes: np.ndarray`

- [ ] **Step 1: Write the failing tests**

`tests/models/test_models.py`:

```python
import numpy as np
import optuna
import pandas as pd
import pytest

from copra_grading.models import MODEL_NAMES, fit_model, suggest_params


def _data(classes=("A", "B", "C", "D", "E"), n=40, seed=0, sizes=None):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, c in enumerate(classes):
        k = sizes[c] if sizes else n
        X.append(rng.normal(i * 4.0, 0.5, size=(k, 3)))
        y += [c] * k
    return pd.DataFrame(np.vstack(X), columns=["f1", "f2", "f3"]), pd.Series(y)


def _default_params(name):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return suggest_params(name, study.ask())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_class_letters_for_every_model(name):
    X, y = _data()
    model = fit_model(name, _default_params(name), X, y, seed=0)
    pred = model.predict(X)
    assert set(pred) <= {"A", "B", "C", "D", "E"}
    assert (pred == y.to_numpy()).mean() > 0.9


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_with_five_classes_predicts_string_labels(name):
    # class F missing entirely: model still trains on what is there
    X, y = _data(classes=("A", "C", "E"))
    model = fit_model(name, _default_params(name), X, y, seed=0)
    assert list(model.classes) == ["A", "C", "E"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_regularization_params_are_tuned(name):
    params = _default_params(name)
    required = {
        "random_forest": {"max_depth", "min_samples_leaf"},
        "xgboost": {"max_depth", "min_child_weight"},
        "lightgbm": {"max_depth", "min_child_weight"},
        "logistic_regression": {"C", "l1_ratio"},
    }[name]
    assert required <= set(params)


def test_class_weighting_lifts_minority_recall():
    # 200 A vs 10 B, overlapping: unweighted RF would mostly ignore B
    rng = np.random.default_rng(1)
    X = pd.DataFrame(np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(1.0, 1, (10, 2))]),
                     columns=["f1", "f2"])
    y = pd.Series(["A"] * 200 + ["B"] * 10)
    model = fit_model("xgboost", {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
                                  "min_child_weight": 1.0, "subsample": 1.0,
                                  "colsample_bytree": 1.0, "reg_alpha": 0.0,
                                  "reg_lambda": 1.0}, X, y, seed=0)
    assert (model.predict(X) == "B").sum() >= 10


def test_logistic_regression_is_scaled():
    X, y = _data()
    model = fit_model("logistic_regression", _default_params("logistic_regression"), X, y, seed=0)
    assert model.estimator.steps[0][0] == "scale"


def test_sklearn_models_use_balanced_class_weight():
    X, y = _data()
    for name in ("random_forest", "lightgbm"):
        model = fit_model(name, _default_params(name), X, y, seed=0)
        assert model.estimator.get_params()["class_weight"] == "balanced"
    lr = fit_model("logistic_regression", _default_params("logistic_regression"), X, y, seed=0)
    assert lr.estimator.named_steps["model"].class_weight == "balanced"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_models.py -q`
Expected: FAIL with `ImportError: cannot import name 'MODEL_NAMES'`

- [ ] **Step 3: Implement the four model modules**

`src/copra_grading/models/logreg.py` (replace body; keep the module docstring, update its last line):

```python
"""Logistic Regression - baseline only, never eligible for deployment. process.md Step 5.

Requires standardized feature scaling (only model in this pipeline that does -
see ../../../CLAUDE.md). The scaler sits inside the pipeline, so it is fitted
on each training split only.
"""

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space. `penalty` l1/l2 is l1_ratio 1.0/0.0 (sklearn 1.9)."""
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
    }


def build_model(params: dict, seed: int) -> Pipeline:
    """StandardScaler + class_weight='balanced' Logistic Regression."""
    model = LogisticRegression(
        solver="saga", max_iter=5000, class_weight="balanced", random_state=seed, **params
    )
    return Pipeline([("scale", StandardScaler()), ("model", model)])
```

`src/copra_grading/models/random_forest.py`:

```python
"""Random Forest ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

import optuna
from sklearn.ensemble import RandomForestClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def build_model(params: dict, seed: int) -> RandomForestClassifier:
    """Construct a class_weight='balanced' Random Forest from tuned params."""
    return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
```

`src/copra_grading/models/xgboost_model.py`:

```python
"""XGBoost ensemble candidate. process.md Step 5. No feature scaling needed (tree-based).

XGBoost has no class_weight argument: models.fit_model passes balanced
per-row sample weights instead.
"""

import optuna
from xgboost import XGBClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_model(params: dict, seed: int) -> XGBClassifier:
    """Construct an XGBoost classifier; labels must be integer-encoded."""
    return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1, **params)
```

`src/copra_grading/models/lightgbm_model.py`:

```python
"""LightGBM ensemble candidate. process.md Step 5. No feature scaling needed (tree-based)."""

import optuna
from lightgbm import LGBMClassifier


def suggest_params(trial: optuna.Trial) -> dict:
    """ADR-003 space, plus max_depth and min_child_weight (process.md)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_model(params: dict, seed: int) -> LGBMClassifier:
    """Construct a class_weight='balanced' LightGBM classifier from tuned params."""
    return LGBMClassifier(
        class_weight="balanced", subsample_freq=1, random_state=seed,
        n_jobs=-1, verbose=-1, **params,
    )
```

- [ ] **Step 4: Implement the registry**

Append to `src/copra_grading/models/__init__.py` (keep the docstring):

```python
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from copra_grading.models import lightgbm_model, logreg, random_forest, xgboost_model

_MODULES = {
    "logistic_regression": logreg,
    "random_forest": random_forest,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}
MODEL_NAMES = tuple(_MODULES)


@dataclass(frozen=True)
class FittedModel:
    """A trained estimator plus the class letters behind its integer labels."""

    name: str
    estimator: object
    encoder: LabelEncoder

    @property
    def classes(self) -> np.ndarray:
        return self.encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.encoder.inverse_transform(self.estimator.predict(X).astype(int))


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    return _MODULES[name].suggest_params(trial)


def fit_model(name: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    """Fit one model on integer-encoded labels. XGBoost gets balanced
    per-row sample weights; the others carry class_weight='balanced'."""
    encoder = LabelEncoder().fit(y)
    y_int = encoder.transform(y)
    estimator = _MODULES[name].build_model(params, seed)
    if name == "xgboost":
        estimator.fit(X, y_int, sample_weight=compute_sample_weight("balanced", y_int))
    else:
        estimator.fit(X, y_int)
    return FittedModel(name, estimator, encoder)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_models.py -q`
Expected: `15 passed` (3 tests x 4 models + 3 single tests)

- [ ] **Step 6: Stage**

```bash
git add src/copra_grading/models tests/models/test_models.py
```

---

### Task 2: Inner-CV tuning

**Files:**
- Modify: `src/copra_grading/models/tuning.py`
- Test: `tests/models/test_tuning.py`

**Interfaces:**
- Consumes: `models.fit_model`, `models.suggest_params`, `folds.make_folds(table, config) -> list[Fold]`, `features.table.feature_columns(table)`, fixtures `feature_table`, `config`.
- Produces: `tune_model(model_name: str, table: pd.DataFrame, config: dict, n_trials: int, seed: int) -> optuna.Study`. `table` is a feature table holding **all 8 transform rows** of the photos tuning may use (the outer training photos). `study.best_params` feeds `fit_model`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_tuning.py`:

```python
from copra_grading.models.tuning import inner_config, tune_model


def test_returns_study_with_macro_f1_value(feature_table, config):
    study = tune_model("random_forest", feature_table, config, n_trials=2, seed=0)
    assert len(study.trials) == 2
    assert 0.0 <= study.best_value <= 1.0
    assert {"max_depth", "min_samples_leaf"} <= set(study.best_params)


def test_same_seed_same_best_params(feature_table, config):
    a = tune_model("lightgbm", feature_table, config, n_trials=3, seed=7)
    b = tune_model("lightgbm", feature_table, config, n_trials=3, seed=7)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value


def test_inner_config_uses_inner_split_count(config):
    inner = inner_config(config)
    assert inner["splitting"]["n_splits"] == 3
    assert config["splitting"]["n_splits"] == 5  # caller's config untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_tuning.py -q`
Expected: FAIL with `ImportError: cannot import name 'inner_config'`

- [ ] **Step 3: Implement**

Replace `src/copra_grading/models/tuning.py`:

```python
"""Optuna hyperparameter tuning. process.md Step 5. Search spaces/budget: see ADR-003.

Objective is Macro F1 for every model, not raw accuracy - plain accuracy
would let the majority classes among A-F dominate the score.

Nested CV: tune_model only ever receives the photos of one outer training
fold. It re-splits them with folds.make_folds (grouped, stratified, split
before augment) into tuning.inner_n_splits inner folds and scores each trial
on the inner validation originals.
"""

import copy

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.features.table import feature_columns
from copra_grading.folds import make_folds
from copra_grading.models import fit_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_config(config: dict) -> dict:
    """Copy of config with the inner split count."""
    inner = copy.deepcopy(config)
    inner["splitting"]["n_splits"] = config["tuning"]["inner_n_splits"]
    return inner


def tune_model(
    model_name: str,
    table: pd.DataFrame,
    config: dict,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Run an Optuna study for the named model, maximising mean Macro F1
    over grouped + stratified inner folds of `table` (never ungrouped CV -
    leaks parts of one whole sample)."""
    label = config["splitting"]["label_column"]
    features = feature_columns(table)
    inner_folds = make_folds(table, inner_config(config))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        scores = []
        for fold in inner_folds:
            model = fit_model(model_name, params, fold.train[features], fold.train[label], seed)
            pred = model.predict(fold.val[features])
            scores.append(f1_score(fold.val[label], pred, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_tuning.py -q`
Expected: `3 passed`

- [ ] **Step 5: Stage**

```bash
git add src/copra_grading/models/tuning.py tests/models/test_tuning.py
```

---

### Task 3: Outer nested-CV loop and selection

**Files:**
- Modify: `src/copra_grading/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `tune_model`, `fit_model`, `make_folds`, `feature_columns`.
- Produces:
  - `CVResult` dataclass: `scores: dict[str, list[float]]` (model -> Macro F1 per outer fold), `best_params: dict[str, list[dict]]` (model -> per-fold params), `predictions: pd.DataFrame` with columns `id, batch_id, copra_class, fold, model, predicted`.
  - `cross_validate_models(table: pd.DataFrame, config: dict, model_names: tuple[str, ...], n_trials: int) -> CVResult`. `table` = full cleaned feature table (`features_clean.csv`).
  - `select_model(fold_scores: dict[str, list[float]]) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_selection.py`:

```python
import pytest

import copra_grading.selection as selection
from copra_grading.selection import cross_validate_models, select_model


def test_select_model_ignores_logistic_regression():
    scores = {"logistic_regression": [0.99] * 5, "random_forest": [0.7] * 5,
              "xgboost": [0.8] * 5, "lightgbm": [0.75] * 5}
    assert select_model(scores) == "xgboost"


def test_select_model_needs_an_ensemble():
    with pytest.raises(ValueError, match="ensemble"):
        select_model({"logistic_regression": [0.9] * 5})


def test_every_photo_predicted_once_per_model(feature_table, config):
    result = cross_validate_models(
        feature_table, config, ("logistic_regression", "random_forest"), n_trials=1
    )
    originals = feature_table[feature_table["augment"] == "rot0"]
    for name in ("logistic_regression", "random_forest"):
        preds = result.predictions[result.predictions["model"] == name]
        assert sorted(preds["id"]) == sorted(originals["id"])
        assert len(result.scores[name]) == 5
        assert len(result.best_params[name]) == 5
    assert set(result.predictions["predicted"]) <= set("ABCDE")


def test_tuning_never_sees_outer_validation_photos(feature_table, config, monkeypatch):
    seen = []
    real = selection.tune_model

    def spy(name, table, cfg, n_trials, seed):
        seen.append(set(table["id"]))
        return real(name, table, cfg, n_trials, seed)

    monkeypatch.setattr(selection, "tune_model", spy)
    result = cross_validate_models(feature_table, config, ("random_forest",), n_trials=1)
    preds = result.predictions
    for k, tuned_ids in enumerate(seen):
        val_ids = set(preds.loc[preds["fold"] == k, "id"])
        assert tuned_ids.isdisjoint(val_ids)
        # tuning gets every transform of its photos, so inner folds can augment
        assert len(feature_table[feature_table["id"].isin(tuned_ids)]) == 8 * len(tuned_ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_selection.py -q`
Expected: FAIL with `ImportError: cannot import name 'tune_model' from 'copra_grading.selection'` (or `NotImplementedError`)

- [ ] **Step 3: Implement**

Replace `src/copra_grading/selection.py`:

```python
"""Model selection. process.md Step 6.

Compare all four models on Macro F1 across the 5 StratifiedGroupKFold folds. The best
of Random Forest / XGBoost / LightGBM is the selected (deployed) model.
Logistic Regression is a reference point only, never selectable.

Nested CV: for each outer fold, every model is tuned on that fold's training
photos only (tuning.tune_model), refitted on the fold's augmented training
rows with the best params, and scored on the fold's validation originals.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from copra_grading.features.table import feature_columns
from copra_grading.folds import make_folds
from copra_grading.models import fit_model
from copra_grading.models.tuning import tune_model

ENSEMBLE_CANDIDATES = ("random_forest", "xgboost", "lightgbm")


@dataclass(frozen=True)
class CVResult:
    scores: dict[str, list[float]]
    best_params: dict[str, list[dict]]
    predictions: pd.DataFrame  # id, batch_id, copra_class, fold, model, predicted


# --- selection ---


def cross_validate_models(
    table: pd.DataFrame, config: dict, model_names: tuple[str, ...], n_trials: int
) -> CVResult:
    """Nested CV over the outer folds of `table` (a cleaned feature table
    with every transform row). Return per-fold Macro F1, per-fold best
    params and out-of-fold predictions per model.
    """
    label = config["splitting"]["label_column"]
    group = config["splitting"]["group_column"]
    seed = config["tuning"]["random_state"]
    features = feature_columns(table)
    scores = {name: [] for name in model_names}
    best_params = {name: [] for name in model_names}
    predictions = []

    for fold in make_folds(table, config):
        train_photos = table[table["id"].isin(fold.train["id"])]
        for name in model_names:
            study = tune_model(name, train_photos, config, n_trials, seed + fold.index)
            model = fit_model(name, study.best_params, fold.train[features],
                              fold.train[label], seed + fold.index)
            pred = model.predict(fold.val[features])
            scores[name].append(
                float(f1_score(fold.val[label], pred, average="macro", zero_division=0))
            )
            best_params[name].append(study.best_params)
            predictions.append(
                fold.val[["id", group, label]].assign(fold=fold.index, model=name, predicted=pred)
            )
            print(f"fold {fold.index} {name}: macro F1 {scores[name][-1]:.3f}", flush=True)

    return CVResult(scores, best_params, pd.concat(predictions, ignore_index=True))


def select_model(fold_scores: dict[str, list[float]]) -> str:
    """Return the ensemble in ENSEMBLE_CANDIDATES with the highest mean Macro F1."""
    candidates = {n: np.mean(s) for n, s in fold_scores.items() if n in ENSEMBLE_CANDIDATES}
    if not candidates:
        raise ValueError("no ensemble model in fold_scores - Logistic Regression is never selectable")
    return max(candidates, key=candidates.get)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_selection.py -q`
Expected: `4 passed`

- [ ] **Step 5: Stage**

```bash
git add src/copra_grading/selection.py tests/test_selection.py
```

---

### Task 4: Step 7 metrics

**Files:**
- Modify: `src/copra_grading/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `dataset.MOISTURE_ORDER = ("F", "A", "B", "C", "D", "E")`.
- Produces: `compute_metrics(y_true, y_pred) -> dict` with keys `labels: list[str]` (moisture order, only classes present in either), `confusion_matrix: list[list[int]]` (rows true, columns predicted), `accuracy: float`, `macro_f1: float`, `per_class: dict[str, dict]` with `precision`, `recall`, `support`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluation.py`:

```python
import pandas as pd
import pytest

from copra_grading.evaluation import compute_metrics


def test_labels_follow_moisture_order_not_letters():
    m = compute_metrics(pd.Series(["E", "A", "F", "B"]), pd.Series(["E", "A", "F", "B"]))
    assert m["labels"] == ["F", "A", "B", "E"]
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_confusion_matrix_rows_true_columns_predicted():
    m = compute_metrics(pd.Series(["A", "A", "B"]), pd.Series(["A", "B", "B"]))
    assert m["confusion_matrix"] == [[1, 1], [0, 1]]
    assert m["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert m["per_class"]["B"]["precision"] == pytest.approx(0.5)


def test_class_never_predicted_has_zero_recall():
    m = compute_metrics(pd.Series(["A", "C", "C"]), pd.Series(["A", "A", "A"]))
    assert m["labels"] == ["A", "C"]
    assert m["per_class"]["C"] == {"precision": 0.0, "recall": 0.0, "support": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -q`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

Replace the body of `compute_metrics` in `src/copra_grading/evaluation.py` and add imports:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from copra_grading.dataset import MOISTURE_ORDER

# --- metrics ---


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return confusion matrix, accuracy, Macro F1, and per-class precision
    and recall. Order confusion-matrix rows/columns by dataset.MOISTURE_ORDER
    so errors between neighbouring moisture bands sit next to the diagonal.
    """
    present = set(y_true) | set(y_pred)
    labels = [c for c in MOISTURE_ORDER if c in present]
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p), "recall": float(r), "support": int(s)}
            for c, p, r, s in zip(labels, precision, recall, support)
        },
    }
```

Note: `macro_f1` here averages over every class present in either series, so a class that is predicted but never true counts as a zero. The per-fold selection score in `selection.py` uses sklearn's default (classes in either series too), so the two agree.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -q`
Expected: `3 passed`

- [ ] **Step 5: Stage**

```bash
git add src/copra_grading/evaluation.py tests/test_evaluation.py
```

---

### Task 5: Training script and real run

**Files:**
- Create: `scripts/train_models.py`
- Modify: `README.md` (usage line)

**Interfaces:**
- Consumes: `cross_validate_models`, `select_model`, `compute_metrics`, `MODEL_NAMES`.
- Produces: `data/processed/results/` with `cv_scores.csv` (model, fold, macro_f1), `best_params.json`, `oof_predictions.csv`, `metrics.json` (per model, from all out-of-fold predictions), `selection.json` (`{"selected": name, "mean_macro_f1": {...}}`).

- [ ] **Step 1: Write the script**

`scripts/train_models.py`:

```python
"""Run Steps 5-7 on the cleaned feature table from prepare_folds.py:
nested-CV tuning and training of all four models, selection of the best
ensemble, and per-photo metrics on the held-out folds.

Usage:
  uv run python scripts/train_models.py                        # all models, config trials
  uv run python scripts/train_models.py --n-trials 3 --models random_forest   # smoke run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from copra_grading.evaluation import compute_metrics
from copra_grading.features.table import ID_COLUMNS
from copra_grading.models import MODEL_NAMES
from copra_grading.selection import ENSEMBLE_CANDIDATES, cross_validate_models, select_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    n_trials = args.n_trials or config["tuning"]["n_trials"]
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    args.out.mkdir(parents=True, exist_ok=True)

    result = cross_validate_models(table, config, tuple(args.models), n_trials)

    scores = pd.DataFrame(
        [(m, k, s) for m, fold_scores in result.scores.items() for k, s in enumerate(fold_scores)],
        columns=["model", "fold", "macro_f1"],
    )
    scores.to_csv(args.out / "cv_scores.csv", index=False)
    (args.out / "best_params.json").write_text(json.dumps(result.best_params, indent=2))
    result.predictions.to_csv(args.out / "oof_predictions.csv", index=False)

    label = config["splitting"]["label_column"]
    metrics = {}
    for name, preds in result.predictions.groupby("model"):
        metrics[name] = compute_metrics(preds[label], preds["predicted"])
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = scores.groupby("model")["macro_f1"].agg(["mean", "std"])
    print(summary.round(3).to_string())
    if any(m in ENSEMBLE_CANDIDATES for m in args.models):
        selected = select_model(result.scores)
        (args.out / "selection.json").write_text(json.dumps(
            {"selected": selected, "mean_macro_f1": summary["mean"].round(4).to_dict()}, indent=2
        ))
        print(f"selected: {selected}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on real data**

Run: `uv run python scripts/train_models.py --n-trials 2 --out data/processed/results_smoke`
Expected: 20 `fold k model: macro F1 ...` lines, a mean/std table for 4 models, `selected: <ensemble>`. Note the wall time; multiply by 75/2 for the full-run estimate.

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q`
Expected: all pass (95 existing + new).

- [ ] **Step 4: Full run (background, about 1-1.5 h)**

Run: `uv run python scripts/train_models.py`
Expected: `data/processed/results/` holds the five files. Check `best_params.json`: if a model's params sit on a range edge in most folds, ADR-003 says widen that range.

- [ ] **Step 5: README usage**

Under the `extract_features.py` usage line in `README.md` add:

```
uv run python scripts/prepare_folds.py      # -> features_clean.csv + folds.csv (outliers, folds)
uv run python scripts/train_models.py       # -> data/processed/results/ (nested CV, selection, metrics)
```

and remove "models + tuning, selection, evaluation" from the `NotImplementedError` list in the status paragraph.

- [ ] **Step 6: Stage**

```bash
git add scripts/train_models.py README.md
```

Report the per-model mean ± std Macro F1, the selected model, and the selected model's confusion matrix to the user. Class D numbers carry the ADR-011 lighting caveat. Leave committing to the user.

---

## Out of scope (separate plans)

- Final deployable model: tune once on all cleaned photos and refit (Step 9 artifact, ADR-005).
- TreeSHAP by feature group (Step 8), Streamlit app (Step 9).
- Outlier rule per class (option 2 from the 2026-10-14 outlier review) - needs its own ADR first.
