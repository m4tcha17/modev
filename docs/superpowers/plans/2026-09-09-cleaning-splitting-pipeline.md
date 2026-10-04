# Cleaning + Splitting Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flatten the one-directory-per-stage scaffold to one module file per stage, then implement duplicate `Sample_ID` detection, feature-level outlier flagging, and GroupKFold splitting.

**Architecture:** Part 1 is a mechanical restructure with zero behavior change — collapse 7 stub packages into single `.py` files, fix imports, update docs, prove the 25-test suite still passes. Part 2 fills `cleaning.py` and `splitting.py` with pure `pandas`/`numpy`/`scikit-learn` functions, TDD, validated against synthetic fixtures (no dataset exists yet). Cleaning detects and reports conflicts but never drops or mutates rows; outlier flagging is advisory (returns a boolean Series); splitting wraps `GroupKFold` and runs before any augmentation.

**Tech Stack:** Python ≥3.10, pandas ≥2.2, numpy ≥1.26, scikit-learn ≥1.4, pytest ≥8.0, `uv` for env management.

**Spec:** `docs/superpowers/specs/2026-09-09-cleaning-splitting-design.md`

## Global Constraints

- **NEVER run `git commit`.** Every task ends with `git add` of the task's files only. The user commits manually at review checkpoints. Subagents and executors: stage, never commit. (`CLAUDE.md` Git section; memory `feedback-no-auto-commit`.)
- **Run tests with:** `uv run --extra dev python -m pytest -q` (bare `python -m pytest` fails — `cv2` only resolves inside the uv env).
- **Classification only.** No regression head, no commercial grade names (`Resecada`/`Bodega`/`Corriente`) anywhere in code, tests, or output. Not exercised by this stage; do not introduce.
- **Split before augment.** `splitting.py` produces folds; augmentation happens later, only inside each fold's training side. Never split after augmenting.
- **No row dropping / imputation in cleaning.** Completeness is guaranteed upstream by Jotter. Cleaning detects and reports; it does not fix.
- **ADR-001 outlier method (IQR, 1.5×) is a default pending thesis-team confirmation** — say so in the `cleaning.py` module docstring and `README.md`, do not present it as final.
- **Keep as packages, do not collapse:** `features/`, `models/`, `preprocessing/` (implemented and/or genuinely multi-part).

---

## Task 1: Flatten stub packages to one file per stage

**Files:**
- Create: `src/copra_grading/cleaning.py`, `src/copra_grading/splitting.py`, `src/copra_grading/augmentation.py`, `src/copra_grading/selection.py`, `src/copra_grading/explainability.py`, `src/copra_grading/evaluation.py`, `src/copra_grading/artifact.py`
- Delete: the directories `src/copra_grading/{cleaning,splitting,augmentation,selection,explainability,evaluation,artifact}/` (all their `.py` files)
- Delete: empty test dirs `tests/{cleaning,splitting,augmentation,selection,explainability,evaluation,artifact}/` (each holds only `.gitkeep`)
- Modify: `docs/architecture.md` (module→section table, paths only)
- Modify: `docs/decisions/ADR-001-outlier-detection-method.md`, `ADR-002-augmentation-multiplier.md`, `ADR-005-artifact-serialization-format.md`, `ADR-007-boundary-tolerance-band.md` (path references only)

**Interfaces:**
- Consumes: nothing.
- Produces: import paths change from `copra_grading.cleaning.dedup` → `copra_grading.cleaning`, `copra_grading.splitting.groupkfold` → `copra_grading.splitting`, and likewise for `augmentation`, `selection`, `explainability`, `evaluation`, `artifact`. All existing public function names and signatures are preserved unchanged (still `raise NotImplementedError` — Task 2+ replace `cleaning.py` and `splitting.py`). There are currently **no** cross-package imports among these seven (verified), so this is pure file concatenation.

- [ ] **Step 1: Confirm the starting state is green**

Run: `uv run --extra dev python -m pytest -q`
Expected: `25 passed`.

- [ ] **Step 2: Create `src/copra_grading/cleaning.py`**

Merge the three files' bodies into one module. Module docstring = the old `cleaning/__init__.py` docstring. Then a section banner and each function, still as stubs:

```python
"""Duplicate Sample_ID resolution + feature-level outlier detection. Spec §6.

Completeness (all six images + moisture reading present) is guaranteed
upstream by Jotter - no imputation logic belongs here.

Outlier method/cutoff (IQR, 1.5x) is an ADR-001 default pending thesis-team
confirmation, not a fixed decision - see README.
"""

import pandas as pd

# --- §6.1 Duplicate Sample_ID resolution --------------------------------------


def resolve_duplicate_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and resolve duplicate Sample_ID rows across the merged dataset."""
    raise NotImplementedError


# --- §6.2 Feature-level outlier detection (ADR-001) --------------------------


def flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    """Flag rows with any feature value outside Q1 - m*IQR .. Q3 + m*IQR."""
    raise NotImplementedError


def flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flag rows with any feature |z-score| above threshold."""
    raise NotImplementedError
```

Then delete the `src/copra_grading/cleaning/` directory.

- [ ] **Step 3: Create the other six modules the same way**

For each package, new `src/copra_grading/<stage>.py` = the `__init__.py` docstring as module docstring, then one `# --- <sub-section> ---` banner per former file with that file's imports (de-duplicated at the top of the module) and function stubs verbatim. Then delete the directory.

- `splitting.py` ← `splitting/__init__.py` + `splitting/groupkfold.py` (`split_by_sample`)
- `augmentation.py` ← `augmentation/__init__.py` + `augmentation/geometric.py` (`augment_sample`)
- `selection.py` ← `selection/__init__.py` + `selection/algorithm_selection.py` (`select_algorithm`) + `selection/deployment_config.py` (`evaluate_angle_configs`, `choose_deployment_config`)
- `explainability.py` ← `explainability/__init__.py` + `explainability/treeshap.py` (`compute_aggregate_shap_by_family`, `explain_single_prediction`)
- `evaluation.py` ← `evaluation/__init__.py` + `metrics.py` (`compute_metrics`) + `boundary_analysis.py` (`bucket_by_boundary_proximity`, keeps `from copra_grading.labels import LOWER_THRESHOLD, UPPER_THRESHOLD`) + `human_baseline.py` (`compare_to_human_baseline`) + `ablation.py` (`run_ablation`) + `viz.py` (`plot_feature_space`)
- `artifact.py` ← `artifact/__init__.py` + `artifact/serialize.py` (`save_artifact`, `load_artifact`, keeps `from pathlib import Path`)

- [ ] **Step 4: Remove the empty test directories**

```bash
git rm -r tests/cleaning tests/splitting tests/augmentation tests/selection tests/explainability tests/evaluation tests/artifact
```

(These contain only `.gitkeep`. `tests/features/`, `tests/models/`, `tests/preprocessing/` stay.)

- [ ] **Step 5: Update `docs/architecture.md`**

In the "Module → spec section map" table, rewrite the left column:
`cleaning/dedup.py` and `cleaning/outliers.py` → collapse to one row `cleaning.py` (§6); `splitting/groupkfold.py` → `splitting.py`; `augmentation/geometric.py` → `augmentation.py`; `selection/algorithm_selection.py` + `selection/deployment_config.py` → two rows both `selection.py`; `explainability/treeshap.py` → `explainability.py`; the five `evaluation/*.py` rows → all `evaluation.py`; `artifact/serialize.py` → `artifact.py`. Leave `labels.py`, `preprocessing/otsu.py`, `features/*` rows unchanged.

- [ ] **Step 6: Update ADR path references**

- `ADR-001` line 10: `` `cleaning/outliers.py` `` → `` `cleaning.py` ``
- `ADR-002` line 14: `` `augmentation/geometric.py` `` → `` `augmentation.py` ``
- `ADR-005` lines 8, 12: `` `artifact/serialize.py` `` → `` `artifact.py` ``
- `ADR-007` line 8: `` `evaluation/boundary_analysis.py` `` → `` `evaluation.py` ``

- [ ] **Step 7: Run the full suite — must be unchanged**

Run: `uv run --extra dev python -m pytest -q`
Expected: `25 passed`. If anything errors on import, a collapsed module is missing a symbol or an import line — fix the module, not the test.

- [ ] **Step 8: Verify the package still imports cleanly**

Run: `uv run python -c "import copra_grading.cleaning, copra_grading.splitting, copra_grading.augmentation, copra_grading.selection, copra_grading.explainability, copra_grading.evaluation, copra_grading.artifact; print('ok')"`
Expected: `ok`

- [ ] **Step 9: Stage**

```bash
git add -A src/copra_grading tests docs/architecture.md docs/decisions
git status
```

Report to the reviewer: "Part 1 restructure staged, 25 tests still pass, no behavior change. Please commit before Part 2."

---

## Task 2: `cleaning.py` — duplicate `Sample_ID` detection (§6.1)

**Files:**
- Modify: `src/copra_grading/cleaning.py` (replace the `resolve_duplicate_samples` stub and its section)
- Test: `tests/test_cleaning.py` (create)

**Interfaces:**
- Consumes: nothing (operates on a caller-supplied DataFrame).
- Produces:
  - `ANGLES_PER_SAMPLE: int = 6`
  - `@dataclass class SampleConflict` with fields `sample_id: str`, `reason: str` (`"row_count"` | `"duplicate_angle"` | `"moisture_conflict"`), `detail: str`, `row_labels: list` (index labels of the offending rows in the input frame)
  - `@dataclass class DedupReport` with field `conflicts: list[SampleConflict]` and property `is_clean: bool` (`True` iff `conflicts` is empty)
  - `class DuplicateSampleError(Exception)` — constructed as `DuplicateSampleError(report: DedupReport)`, stores `.report`, message lists one line per conflict
  - `resolve_duplicate_samples(df: pd.DataFrame) -> DedupReport`
  - `assert_clean(df: pd.DataFrame) -> None` — raises `DuplicateSampleError` when not clean, else returns `None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cleaning.py`:

```python
import numpy as np
import pandas as pd
import pytest

from copra_grading.cleaning import (
    DedupReport,
    DuplicateSampleError,
    SampleConflict,
    assert_clean,
    resolve_duplicate_samples,
)


def _sample_rows(sample_id, moisture, n_angles=6):
    return pd.DataFrame(
        {
            "Sample_ID": [sample_id] * n_angles,
            "Angle_ID": list(range(1, n_angles + 1)),
            "moisture_reading": [moisture] * n_angles,
            "feat_a": np.linspace(0.0, 1.0, n_angles),
        }
    )


def _frame(*frames):
    return pd.concat(frames, ignore_index=True)


def test_clean_frame_reports_no_conflicts():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0))

    report = resolve_duplicate_samples(df)

    assert isinstance(report, DedupReport)
    assert report.is_clean is True
    assert report.conflicts == []


def test_short_sample_flagged_as_row_count_conflict():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    report = resolve_duplicate_samples(df)

    assert report.is_clean is False
    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "row_count") in reasons


def test_duplicate_angle_flagged():
    bad = _sample_rows("S2", 20.0)
    bad.loc[bad.index[-1], "Angle_ID"] = 1  # angle 1 now appears twice
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "duplicate_angle") in reasons


def test_conflicting_moisture_reading_flagged():
    bad = _sample_rows("S2", 20.0)
    bad.loc[bad.index[0], "moisture_reading"] = 4.0  # two physical samples, one ID
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = {(c.sample_id, c.reason) for c in report.conflicts}
    assert ("S2", "moisture_conflict") in reasons


def test_multiple_issues_produce_one_conflict_each():
    bad = _sample_rows("S2", 20.0, n_angles=5)
    bad.loc[bad.index[0], "moisture_reading"] = 4.0
    df = _frame(_sample_rows("S1", 5.0), bad)

    report = resolve_duplicate_samples(df)

    reasons = sorted(c.reason for c in report.conflicts if c.sample_id == "S2")
    assert reasons == ["moisture_conflict", "row_count"]


def test_conflict_carries_row_labels_into_source_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    report = resolve_duplicate_samples(df)
    conflict = next(c for c in report.conflicts if c.sample_id == "S2")

    assert set(conflict.row_labels).issubset(set(df.index))
    assert all(df.loc[label, "Sample_ID"] == "S2" for label in conflict.row_labels)


def test_assert_clean_raises_on_dirty_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0, n_angles=5))

    with pytest.raises(DuplicateSampleError) as excinfo:
        assert_clean(df)

    assert isinstance(excinfo.value.report, DedupReport)
    assert "S2" in str(excinfo.value)


def test_assert_clean_returns_none_on_clean_frame():
    df = _frame(_sample_rows("S1", 5.0), _sample_rows("S2", 20.0))

    assert assert_clean(df) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_cleaning.py -q`
Expected: FAIL — `ImportError` (`DedupReport`, `SampleConflict`, `DuplicateSampleError`, `assert_clean` not defined) or `NotImplementedError`.

- [ ] **Step 3: Implement the dedup section**

In `src/copra_grading/cleaning.py`, replace the `# --- §6.1 ...` section with:

```python
from dataclasses import dataclass, field

import pandas as pd

ANGLES_PER_SAMPLE = 6


@dataclass
class SampleConflict:
    sample_id: str
    reason: str  # "row_count" | "duplicate_angle" | "moisture_conflict"
    detail: str
    row_labels: list = field(default_factory=list)


@dataclass
class DedupReport:
    conflicts: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.conflicts) == 0


class DuplicateSampleError(Exception):
    def __init__(self, report: "DedupReport") -> None:
        self.report = report
        lines = [
            f"  {c.sample_id}: {c.reason} - {c.detail}" for c in report.conflicts
        ]
        super().__init__("duplicate Sample_ID conflicts:\n" + "\n".join(lines))


def resolve_duplicate_samples(df: pd.DataFrame) -> DedupReport:
    """Detect (never resolve destructively) conflicting Sample_ID groups.

    A conflict is reported, one SampleConflict per issue, when a Sample_ID
    group has: a row count != ANGLES_PER_SAMPLE, a repeated Angle_ID, or more
    than one distinct moisture_reading (the offline-clerk ID-collision signal).
    Rows are never dropped or edited here.
    """
    conflicts: list = []
    for sample_id, group in df.groupby("Sample_ID", sort=True):
        sample_id = str(sample_id)
        labels = list(group.index)

        if len(group) != ANGLES_PER_SAMPLE:
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "row_count",
                    f"{len(group)} rows, expected {ANGLES_PER_SAMPLE}",
                    labels,
                )
            )

        dup_mask = group["Angle_ID"].duplicated(keep=False)
        if dup_mask.any():
            repeated = sorted({str(a) for a in group.loc[dup_mask, "Angle_ID"]})
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "duplicate_angle",
                    f"repeated Angle_ID: {', '.join(repeated)}",
                    list(group.index[dup_mask]),
                )
            )

        distinct_moisture = sorted(group["moisture_reading"].dropna().unique())
        if len(distinct_moisture) > 1:
            conflicts.append(
                SampleConflict(
                    sample_id,
                    "moisture_conflict",
                    f"{len(distinct_moisture)} distinct moisture_reading values: {distinct_moisture}",
                    labels,
                )
            )

    return DedupReport(conflicts)


def assert_clean(df: pd.DataFrame) -> None:
    """Raise DuplicateSampleError if resolve_duplicate_samples finds any conflict."""
    report = resolve_duplicate_samples(df)
    if not report.is_clean:
        raise DuplicateSampleError(report)
```

Keep the two `flag_outliers_*` stubs below untouched for now. Ensure only one `import pandas as pd` and one import block at the top of the file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_cleaning.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run --extra dev python -m pytest -q`
Expected: `33 passed` (25 + 8).

- [ ] **Step 6: Stage**

```bash
git add src/copra_grading/cleaning.py tests/test_cleaning.py
git status
```

---

## Task 3: `cleaning.py` — feature-level outlier flagging (§6.2, ADR-001)

**Files:**
- Modify: `src/copra_grading/cleaning.py` (replace the two `flag_outliers_*` stubs, add dispatcher)
- Modify: `configs/default.yaml` (`cleaning:` block — add `outlier_min_features`)
- Test: `tests/test_cleaning.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5, min_flagged_features: int = 3) -> pd.Series` — boolean Series aligned to `features.index`
  - `flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0, min_flagged_features: int = 3) -> pd.Series`
  - `flag_outliers(features: pd.DataFrame, config: dict) -> pd.Series` — reads `config["cleaning"]["outlier_method"]` (`"iqr"`|`"zscore"`), `iqr_multiplier`, `zscore_threshold`, `outlier_min_features` (default 3); raises `ValueError` on an unknown method
  - Identifier columns never counted: `Sample_ID`, `Angle_ID`, `moisture_reading`, plus any non-numeric dtype

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cleaning.py`:

```python
from copra_grading.cleaning import (
    flag_outliers,
    flag_outliers_iqr,
    flag_outliers_zscore,
)


def _feature_frame(n_rows=40, n_feats=10, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.normal(0.0, 1.0, n_rows) for i in range(n_feats)}
    df = pd.DataFrame(data)
    df.insert(0, "Sample_ID", [f"S{r // 6}" for r in range(n_rows)])
    df.insert(1, "Angle_ID", [r % 6 + 1 for r in range(n_rows)])
    df.insert(2, "moisture_reading", 8.0)
    return df


def test_iqr_flags_a_row_with_many_extreme_features():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[3], feat_cols] = 50.0  # every feature wildly out

    flags = flag_outliers_iqr(df, multiplier=1.5, min_flagged_features=3)

    assert flags.loc[df.index[3]] is np.True_ or bool(flags.loc[df.index[3]]) is True
    assert flags.drop(df.index[3]).sum() == 0


def test_min_flagged_features_threshold_respected():
    df = _feature_frame()
    df.loc[df.index[5], ["feat_0", "feat_1"]] = 50.0  # exactly 2 features out

    assert bool(flag_outliers_iqr(df, min_flagged_features=3).loc[df.index[5]]) is False
    assert bool(flag_outliers_iqr(df, min_flagged_features=2).loc[df.index[5]]) is True


def test_flags_series_is_index_aligned():
    df = _feature_frame()
    df.index = [f"row-{i}" for i in range(len(df))]

    flags = flag_outliers_iqr(df)

    assert list(flags.index) == list(df.index)
    assert flags.dtype == bool


def test_zscore_also_flags_the_obvious_outlier():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[7], feat_cols] = 40.0

    flags = flag_outliers_zscore(df, threshold=3.0, min_flagged_features=3)

    assert bool(flags.loc[df.index[7]]) is True


def test_identifier_columns_never_contribute():
    df = _feature_frame(n_rows=12)
    df["Angle_ID"] = df["Angle_ID"] * 1000  # large, but must be ignored
    df["moisture_reading"] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 99.0, 4.0, 4.0, 4.0, 4.0, 4.0]

    flags = flag_outliers_iqr(df, min_flagged_features=1)

    assert flags.sum() == 0


def test_zero_variance_feature_column_does_not_crash_or_flag():
    df = _feature_frame()
    df["feat_const"] = 7.0

    flags = flag_outliers_iqr(df, min_flagged_features=1)

    assert flags.sum() == 0


def test_flag_outliers_dispatches_on_config_method():
    df = _feature_frame()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df.loc[df.index[2], feat_cols] = 60.0

    cfg = {
        "cleaning": {
            "outlier_method": "zscore",
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3.0,
            "outlier_min_features": 3,
        }
    }

    flags = flag_outliers(df, cfg)
    assert bool(flags.loc[df.index[2]]) is True


def test_flag_outliers_rejects_unknown_method():
    cfg = {"cleaning": {"outlier_method": "madness", "outlier_min_features": 3}}
    with pytest.raises(ValueError):
        flag_outliers(_feature_frame(), cfg)


def test_all_non_numeric_frame_returns_all_false():
    df = pd.DataFrame({"Sample_ID": ["S1", "S1"], "note": ["a", "b"]})
    flags = flag_outliers_iqr(df)
    assert flags.tolist() == [False, False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_cleaning.py -q`
Expected: FAIL — the new `flag_outliers_*` raise `NotImplementedError`, `flag_outliers` import fails.

- [ ] **Step 3: Implement the outlier section**

In `src/copra_grading/cleaning.py`, add `import numpy as np` to the top import block, then replace the `# --- §6.2 ...` section with:

```python
_IDENTIFIER_COLUMNS = ("Sample_ID", "Angle_ID", "moisture_reading")


def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    kept = [c for c in features.columns if c not in _IDENTIFIER_COLUMNS]
    return features[kept].select_dtypes(include="number")


def _row_flags(out_of_bounds: pd.DataFrame, min_flagged_features: int, index) -> pd.Series:
    counts = out_of_bounds.sum(axis=1)
    return (counts >= min_flagged_features).reindex(index, fill_value=False).astype(bool)


def flag_outliers_iqr(
    features: pd.DataFrame, multiplier: float = 1.5, min_flagged_features: int = 3
) -> pd.Series:
    """Flag a row when >= min_flagged_features of its numeric features fall
    outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]. Advisory only - nothing
    is dropped. Method/cutoff is an ADR-001 default pending confirmation.
    """
    numeric = _numeric_features(features)
    if numeric.empty:
        return pd.Series(False, index=features.index, dtype=bool)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    out_of_bounds = numeric.lt(lower, axis=1) | numeric.gt(upper, axis=1)
    return _row_flags(out_of_bounds, min_flagged_features, features.index)


def flag_outliers_zscore(
    features: pd.DataFrame, threshold: float = 3.0, min_flagged_features: int = 3
) -> pd.Series:
    """Flag a row when >= min_flagged_features of its numeric features have
    |z-score| > threshold. Zero-variance columns contribute nothing.
    """
    numeric = _numeric_features(features)
    if numeric.empty:
        return pd.Series(False, index=features.index, dtype=bool)
    std = numeric.std(ddof=0).replace(0.0, np.nan)
    z = (numeric - numeric.mean()).abs().div(std, axis=1)
    out_of_bounds = z.gt(threshold).fillna(False)
    return _row_flags(out_of_bounds, min_flagged_features, features.index)


def flag_outliers(features: pd.DataFrame, config: dict) -> pd.Series:
    """Dispatch to the method named in config["cleaning"]["outlier_method"]."""
    cleaning_cfg = config["cleaning"]
    method = cleaning_cfg["outlier_method"]
    min_flagged = cleaning_cfg.get("outlier_min_features", 3)
    if method == "iqr":
        return flag_outliers_iqr(
            features, cleaning_cfg["iqr_multiplier"], min_flagged
        )
    if method == "zscore":
        return flag_outliers_zscore(
            features, cleaning_cfg["zscore_threshold"], min_flagged
        )
    raise ValueError(f"unknown outlier_method: {method!r} (expected 'iqr' or 'zscore')")
```

- [ ] **Step 4: Add the config key**

In `configs/default.yaml`, the `cleaning:` block becomes:

```yaml
cleaning:
  outlier_method: "iqr"         # iqr | zscore (ADR-001)
  iqr_multiplier: 1.5
  zscore_threshold: 3.0
  outlier_min_features: 3       # row flagged only if >= N features are out of bounds
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_cleaning.py -q`
Expected: PASS (17 tests total in the file).

- [ ] **Step 6: Run the full suite**

Run: `uv run --extra dev python -m pytest -q`
Expected: `42 passed` (33 + 9).

- [ ] **Step 7: Update README with the ADR-001 caveat**

In `README.md`, wherever the cleaning stage is described (or add a short "Open parameters" note if none exists), state: "Outlier detection defaults to IQR with a 1.5× multiplier and flags a row only when ≥3 features are out of bounds (`cleaning.outlier_min_features`). Method, multiplier, and threshold are ADR-001 defaults the thesis team has not yet confirmed against real feature distributions."

- [ ] **Step 8: Stage**

```bash
git add src/copra_grading/cleaning.py tests/test_cleaning.py configs/default.yaml README.md
git status
```

---

## Task 4: `splitting.py` — GroupKFold by `Sample_ID` (§7a)

**Files:**
- Modify: `src/copra_grading/splitting.py` (replace the `split_by_sample` stub)
- Test: `tests/test_splitting.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `split_by_sample(df: pd.DataFrame, n_splits: int, group_column: str = "Sample_ID") -> list[tuple[np.ndarray, np.ndarray]]` — `n_splits` `(train_idx, val_idx)` pairs of positional row indices, grouped so no `group_column` value crosses folds
  - Raises `ValueError` if `group_column` is absent or `n_splits` exceeds the number of distinct groups

- [ ] **Step 1: Write the failing tests**

Create `tests/test_splitting.py`:

```python
import numpy as np
import pandas as pd
import pytest

from copra_grading.splitting import split_by_sample


def _frame(n_samples=10, n_angles=6):
    rows = []
    for s in range(n_samples):
        for a in range(n_angles):
            rows.append({"Sample_ID": f"S{s}", "Angle_ID": a + 1, "feat": s + a})
    return pd.DataFrame(rows)


def test_returns_n_splits_pairs():
    folds = split_by_sample(_frame(), n_splits=5)

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)


def test_no_sample_id_crosses_a_fold():
    df = _frame()
    for train_idx, val_idx in split_by_sample(df, n_splits=5):
        train_ids = set(df.iloc[train_idx]["Sample_ID"])
        val_ids = set(df.iloc[val_idx]["Sample_ID"])
        assert train_ids.isdisjoint(val_ids)


def test_every_row_in_exactly_one_validation_fold():
    df = _frame()
    seen = np.zeros(len(df), dtype=int)
    for _, val_idx in split_by_sample(df, n_splits=5):
        seen[val_idx] += 1

    assert (seen == 1).all()


def test_more_splits_than_groups_raises():
    df = _frame(n_samples=3)
    with pytest.raises(ValueError):
        split_by_sample(df, n_splits=5)


def test_missing_group_column_raises():
    df = _frame().drop(columns=["Sample_ID"])
    with pytest.raises(ValueError):
        split_by_sample(df, n_splits=3)


def test_custom_group_column():
    df = _frame().rename(columns={"Sample_ID": "batch"})
    folds = split_by_sample(df, n_splits=5, group_column="batch")

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_ids = set(df.iloc[train_idx]["batch"])
        val_ids = set(df.iloc[val_idx]["batch"])
        assert train_ids.isdisjoint(val_ids)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_splitting.py -q`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `split_by_sample`**

In `src/copra_grading/splitting.py`, replace the stub:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_by_sample(
    df: pd.DataFrame, n_splits: int, group_column: str = "Sample_ID"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold split, grouped by group_column, run BEFORE augmentation.

    Every angle image of one physical sample - and every augmented variant
    generated from it later - stays in the same fold. Returns n_splits
    (train_idx, val_idx) pairs of positional row indices.
    """
    if group_column not in df.columns:
        raise ValueError(f"group_column {group_column!r} not in dataframe columns")

    groups = df[group_column].to_numpy()
    n_groups = pd.unique(groups).size
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of distinct "
            f"{group_column} groups ({n_groups})"
        )

    splitter = GroupKFold(n_splits=n_splits)
    return [
        (train_idx, val_idx)
        for train_idx, val_idx in splitter.split(df, groups=groups)
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_splitting.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run --extra dev python -m pytest -q`
Expected: `48 passed` (42 + 6).

- [ ] **Step 6: Stage**

```bash
git add src/copra_grading/splitting.py tests/test_splitting.py
git status
```

Report to the reviewer: "Cleaning + splitting stage complete. 48 tests pass. Nothing committed — stage is `git add`-ed, ready for your commit."

---

## Self-Review

**1. Spec coverage:**
- §6.1 duplicate `Sample_ID` resolution → Task 2 (row_count, duplicate_angle, moisture_conflict; detect+report, no drops)
- §6.2 outlier flagging, ADR-001 (IQR + Z-score behind one interface, default IQR, flagged as unconfirmed) → Task 3
- §7a GroupKFold by `Sample_ID`, before augmentation, pure CV → Task 4
- Part 1 restructure (7 packages collapsed, `features`/`models`/`preprocessing` kept, docs updated) → Task 1
- Config key `outlier_min_features` → Task 3 Step 4
- README ADR-001 caveat → Task 3 Step 7
- No-commit rule → Global Constraints + every task ends at `git add`

**2. Placeholder scan:** No TBD/TODO. Every code step has full code. Test bodies are complete. README edit (Task 3 Step 7) specifies exact wording.

**3. Type consistency:** `DedupReport.is_clean` is a property in the interface block and the implementation. `SampleConflict.row_labels` named consistently (not `row_indices`) in interface, code, and tests. `flag_outliers(features, config)` signature identical in interface, implementation, and `test_flag_outliers_dispatches_on_config_method`. `split_by_sample` signature identical across interface, Task 4 code, and tests. Test counts are cumulative and consistent: 25 → 33 → 42 → 48.

**Note on test truthiness assertions:** pandas `.loc` on a bool Series returns `numpy.bool_`; tests use `bool(...)` wrapping or `is np.True_` fallback to stay robust.
