# ADR-001: Outlier Detection Method and Cutoff

**Status:** resolved by `model-development-instructions.md` §6.2 and acceptance P3

## Decision

Run on extracted feature values, not images. A value is an outlier if **|z| > 3 or** it falls outside the IQR fence `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR`. Any photo row holding at least one outlier value is **dropped**. No imputation. Thresholds are never relaxed; if too much data is lost, collect more data.

Implemented in `cleaning.py` (`remove_outliers`). Zero-variance columns contribute no flags under the rule that degenerates for them (std == 0 for z, IQR == 0 for the fence).

## Superseded

The earlier default (IQR only, advisory flags, row flagged only when ≥3 features were out of bounds) is replaced.

## Whole row, not single value (decided 2026-11-02)

The spec (§6.2, acceptance P3) says "drop flagged values". This repo reads that as dropping the **whole photo row**. Dropping only the flagged value would leave a null in that cell, and the spec forbids filling nulls in (no imputation). Example: a photo has 65 feature values and one is an outlier, so the whole photo, with all 65 values, is removed. Decided by the user on 2026-11-02.

## Reporting (acceptance P3)

Report per grade both the number of flagged values and the number of photos dropped because of them. If dropping pushes any grade below 50 cubes, the answer is more data collection, never relaxed thresholds.

## Watch

With ~70 features and an OR of two rules, the share of rows dropped may be large. `remove_outliers` returns the flag Series so the loss can be reported per run.
