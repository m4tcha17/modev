# ADR-001: Outlier Detection Method and Cutoff

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** §6.2, §15 item 1

## Default

**IQR method**, standard multiplier: flag values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`.

Implemented alongside Z-score (`|z| > 3`) as a configurable alternative behind one interface (`cleaning/outliers.py`) — not a fixed decision.

## Rationale

IQR is more robust to skewed distributions and less sensitive to the very outliers it's trying to detect than Z-score's mean/std, which the outliers themselves distort. Applies to extracted feature values, not raw images or moisture readings (file-level completeness is already guaranteed upstream by Jotter).

Cite: Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.

## What would change it

Thesis team reviews outlier flags on real extracted features and either confirms IQR/1.5×, adjusts the multiplier, or switches to Z-score if the feature distributions turn out closer to normal than expected.

## Owner

Thesis team (data/methodology decision).
