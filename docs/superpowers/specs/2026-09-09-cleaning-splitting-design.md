# Cleaning + Splitting Stage — Design

**Date:** 2026-09-09
**Status:** approved, pending implementation plan
**Spec reference:** `model-development-instructions.md` §6, §7a; `docs/architecture.md` stages D–E; ADR-001
**Depends on:** preprocessing + feature extraction (done, merged `f3843a0` 2026-09-04)

## Goal

Two parts in one branch:

1. **Repo restructure** (no logic change) — collapse the one-directory-per-stage
   scaffold into one module file per stage.
2. **Implement** cleaning + splitting logic in the new flat modules:
   - `cleaning.py` — duplicate `Sample_ID` resolution (§6.1) + feature-level
     outlier flagging (§6.2, ADR-001)
   - `splitting.py` — GroupKFold by `Sample_ID` (§7a)

Stage stops **before** augmentation (`augmentation.py` stays a stub).

No dataset exists yet. All logic is pure `pandas` / `numpy` / `scikit-learn`
functions, validated against synthetic fixtures — same approach used for the
preprocessing/features stage.

## Part 1 — repo restructure

The scaffold put each spec sub-section in its own file: `evaluation/` alone is
6 files totalling 75 LOC (~12 LOC/file). At thesis scale that is navigation
tax with no isolation benefit. Only stubs exist for these stages, and only
`features/`, `preprocessing/`, `labels` have tests — so this is the cheapest
possible moment to flatten.

**Rule: one module file per pipeline stage.** Collapse these packages, each to
a single `.py` with comment-banner sections (`# --- §6.1 Duplicate Sample_ID
resolution ---`):

| Was (package) | Becomes | Sections |
|---|---|---|
| `cleaning/` (3 files) | `cleaning.py` | dedup, outliers |
| `splitting/` (2 files) | `splitting.py` | groupkfold |
| `augmentation/` (2 files) | `augmentation.py` | geometric (stub) |
| `selection/` (3 files) | `selection.py` | algorithm_selection, deployment_config |
| `explainability/` (2 files) | `explainability.py` | treeshap (stub) |
| `evaluation/` (6 files) | `evaluation.py` | metrics, boundary_analysis, human_baseline, ablation, viz |
| `artifact/` (2 files) | `artifact.py` | serialize (stub) |

**Kept as packages (unchanged):**
- `features/` — 196 LOC, genuinely multi-part, implemented and tested. Re-splitting
  done code is churn.
- `models/` — stays `models/` (`logreg.py`, `random_forest.py`, `xgboost_model.py`,
  `lightgbm_model.py`, `tuning.py`). Each model wrapper is independent and grows
  with its Optuna search space; keeping them separate keeps per-model diffs clean.
- `preprocessing/` — implemented and tested; leave it.

Docstrings currently at package `__init__.py` level move to the module docstring
of the collapsed file. Public names that were imported as
`from copra_grading.cleaning.dedup import X` become
`from copra_grading.cleaning import X`.

**Docs to update in the same branch (paths only, no content change):**
- `docs/architecture.md` — the module→spec-section table
- `docs/decisions/ADR-001` (`cleaning/outliers.py` → `cleaning.py`) and any
  other ADR naming a collapsed path

Result: ~34 source files → ~13.

## Part 2 — cleaning + splitting implementation

## Non-goals

- No imputation for missing images/readings — completeness is guaranteed
  upstream by Jotter (§6).
- No augmentation, no feature scaling, no model code.
- No automatic row dropping or row mutation anywhere in this stage.
- No held-out test set — pure k-fold CV (spec says "GroupKFold
  cross-validation").

## Input contract

One row per angle image, as produced by the feature-extraction stage:

| Column | Type | Notes |
|---|---|---|
| `Sample_ID` | str | six rows share one value (one per angle) |
| `Angle_ID` | str/int | distinct within a `Sample_ID` |
| `moisture_reading` | float | identical across the six rows of a sample |
| `<feature columns>` | float | GLCM + HSV/LAB + Canny, ~50+ columns |

Class label is derived from `moisture_reading` by `labels.derive_class_label`,
never read from a column.

## Module designs

All names below live in `cleaning.py` and `splitting.py` (flat modules from
Part 1). Section banners inside `cleaning.py` separate the dedup code from the
outlier code.

### `cleaning.py` — dedup section (§6.1)

```python
def resolve_duplicate_samples(df: pd.DataFrame) -> DedupReport
def assert_clean(df: pd.DataFrame) -> None   # raises DuplicateSampleError
```

Detect-and-report only. Never drops or edits rows. Per `Sample_ID` group,
flag a conflict when any of:

1. **Row count ≠ 6** — missing or excess angle rows.
2. **Duplicate `Angle_ID`** — same angle appears more than once in the group.
3. **Conflicting `moisture_reading`** — more than one distinct value in the
   group. This is the offline-clerk ID-collision signal: two different
   physical samples were given the same `Sample_ID`.

`DedupReport` (dataclass):

```python
@dataclass
class DedupReport:
    conflicts: list[SampleConflict]   # one per (sample_id, reason)
    is_clean: bool                    # == (len(conflicts) == 0)

@dataclass
class SampleConflict:
    sample_id: str
    reason: str                       # "row_count" | "duplicate_angle" | "moisture_conflict"
    detail: str                       # human-readable, e.g. "5 rows, expected 6"
    row_indices: list[int]            # positional indices into df
```

`assert_clean` calls `resolve_duplicate_samples` and raises
`DuplicateSampleError(report)` if `not report.is_clean`. Pipeline callers use
`assert_clean`; a human inspecting a merge uses `resolve_duplicate_samples`
and reads the report.

Expected-row-count (6) is a module constant `ANGLES_PER_SAMPLE = 6`, matching
the spec's fixed six-angle protocol.

### `cleaning.py` — outliers section (§6.2)

```python
def flag_outliers_iqr(features: pd.DataFrame, multiplier: float = 1.5,
                      min_flagged_features: int = 3) -> pd.Series
def flag_outliers_zscore(features: pd.DataFrame, threshold: float = 3.0,
                         min_flagged_features: int = 3) -> pd.Series
def flag_outliers(features: pd.DataFrame, config: dict) -> pd.Series
```

Runs on extracted feature values, not raw images. Returns a boolean
`pd.Series` aligned to the input index — `True` == flagged. **Nothing is
dropped**; downstream decides what to do with the flags.

Algorithm (both methods):

1. Select numeric feature columns only. Explicitly exclude identifier /
   metadata columns: `Sample_ID`, `Angle_ID`, `moisture_reading`, and any
   non-numeric dtype.
2. Per column, compute bounds:
   - IQR: `[Q1 - multiplier*IQR, Q3 + multiplier*IQR]`
   - Z-score: `|(x - mean) / std| > threshold`
   - A zero-variance column (std or IQR == 0) contributes no flags.
3. Per row, count how many feature columns fall outside their bounds.
4. Flag the row when `count >= min_flagged_features`.

Rationale for `min_flagged_features`: with 50+ columns, an any-single-feature
rule flags almost every row (curse of dimensionality). A configurable count
threshold keeps the flag meaningful. Default 3.

`flag_outliers` dispatches on `config["cleaning"]["outlier_method"]`
(`"iqr"` | `"zscore"`), reading `iqr_multiplier`, `zscore_threshold`,
`outlier_min_features` from the `cleaning` config block.

ADR-001 note stays in the module docstring and README: method and cutoff are
a **default pending thesis-team confirmation**, not a fixed decision.

### `splitting.py` (§7a)

```python
def split_by_sample(df: pd.DataFrame, n_splits: int,
                    group_column: str = "Sample_ID"
                    ) -> list[tuple[np.ndarray, np.ndarray]]
```

Thin wrapper over `sklearn.model_selection.GroupKFold`:

1. Raise `ValueError` if `group_column` not in `df.columns`.
2. Let `n_groups = df[group_column].nunique()`. Raise `ValueError` if
   `n_splits > n_groups` (GroupKFold cannot make more folds than groups).
3. Return `[(train_idx, val_idx), ...]`, `n_splits` pairs, each a positional
   `np.ndarray` of row offsets (`GroupKFold().split(df, groups=...)` output
   passed through unchanged).

No shuffle / random_state — `GroupKFold` is deterministic and exposes
neither. Runs before augmentation (hard ordering constraint, `CLAUDE.md`):
every augmented variant of a sample is generated later, inside the training
side of whichever fold the original landed in.

## Config changes

`configs/default.yaml`, existing `cleaning:` block — add one key:

```yaml
cleaning:
  outlier_method: "iqr"          # iqr | zscore (ADR-001)
  iqr_multiplier: 1.5
  zscore_threshold: 3.0
  outlier_min_features: 3        # row flagged if >= N features out of bounds
```

`splitting:` block already has `n_splits: 5` and `group_column: "Sample_ID"` —
no change.

## Data flow

```
merged feature table (N samples * 6 rows)
  -> assert_clean(df)                     # halt on Sample_ID conflicts
  -> flag_outliers(features, config)      # boolean Series, advisory
  -> split_by_sample(df, n_splits=5)      # list of (train_idx, val_idx)
        -> [augmentation stage, next branch]
```

Where the outlier flags are consumed (drop flagged rows, down-weight, or just
report) is deferred to the stage that assembles training data — this module
only produces the flags.

## Error handling

| Condition | Behavior |
|---|---|
| `Sample_ID` conflict, pipeline path | `assert_clean` raises `DuplicateSampleError` carrying the `DedupReport` |
| `Sample_ID` conflict, inspection path | `resolve_duplicate_samples` returns a report with `is_clean == False`; no raise |
| feature frame all non-numeric | `flag_outliers_*` returns an all-`False` Series (nothing to test) |
| zero-variance feature column | column contributes no flags; no divide-by-zero |
| `n_splits > n_groups` | `split_by_sample` raises `ValueError` |
| `group_column` missing | `split_by_sample` raises `ValueError` |

## Testing

New files `tests/test_cleaning.py`, `tests/test_splitting.py`. Synthetic
fixtures only. (Existing empty `tests/cleaning/`, `tests/splitting/` dirs are
removed along with the other collapsed test dirs — none hold tests.)

Part 1 verification: full suite still green (25 passing) after the restructure,
before any Part 2 logic is written — imports updated, nothing else changed.

**dedup:**
- clean 2-sample / 12-row frame → `is_clean` is `True`, no conflicts
- 5-row sample → one `row_count` conflict with the right `sample_id`
- duplicated `Angle_ID` in a sample → `duplicate_angle` conflict
- two distinct `moisture_reading` values for one `Sample_ID` → `moisture_conflict`
- multiple independent conflicts → one `SampleConflict` entry per issue
- `assert_clean` raises `DuplicateSampleError` on a dirty frame, returns `None` on a clean one

**outliers:**
- injected extreme row (many features far out) → flagged; ordinary rows → not flagged
- `min_flagged_features` respected: a row with 2 wild features is not flagged at default 3, is flagged at 2
- returned Series index matches input index exactly
- IQR and Z-score paths both run and both flag the obvious outlier
- identifier columns (`Sample_ID`, `Angle_ID`, `moisture_reading`) never contribute
- zero-variance column present → no crash, no spurious flags
- `flag_outliers` dispatch picks the method named in config

**split:**
- no `Sample_ID` appears in both train and val of any fold
- every row is in exactly one val fold across all folds
- number of pairs == `n_splits`
- `n_splits > n_groups` → `ValueError`
- missing `group_column` → `ValueError`

Run: `uv run --extra dev python -m pytest -q` (full suite, expect prior 25 + new).

## Size estimate

- Part 1: ~7 package→file collapses (models/, features/, preprocessing/ kept),
  ~15 empty test-dir removals, import fixups, `architecture.md` + ADR path
  edits. No logic change.
- Part 2: `cleaning.py` + `splitting.py` implementations, config key,
  `~12` new tests.

One reviewable branch, Part 1 landing as its own commit(s) before Part 2 so
the restructure diff stays separate from the logic diff.

## Constraints honored

- No `git commit` — staging only, user commits (`CLAUDE.md` Git section,
  `[[feedback-no-auto-commit]]`).
- Split before augment (`CLAUDE.md` hard ordering constraint).
- Classification only, no regression, no grade names — not touched by this
  stage but no code introduces them.
- ADR-001 default (IQR / 1.5×) used, flagged as unconfirmed in README + docstring.
