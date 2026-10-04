# ADR-001: Outlier Detection Method and Cutoff

**Status:** resolved by `process.md` Step 4a (2026-10-04)

## Decision

Run on extracted feature values, not images. A value is an outlier if **|z| > 3 or** it falls outside the IQR fence `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR`. Any photo row holding at least one outlier value is **dropped**. No imputation. Thresholds are never relaxed; if too much data is lost, collect more data.

Implemented in `cleaning.py` (`remove_outliers`). Zero-variance columns contribute no flags under the rule that degenerates for them (std == 0 for z, IQR == 0 for the fence).

## Superseded

The earlier default (IQR only, advisory flags, row flagged only when ≥3 features were out of bounds) is replaced.

## Watch

With ~70 features and an OR of two rules, the share of rows dropped may be large. `remove_outliers` returns the flag Series so the loss can be reported per run.
