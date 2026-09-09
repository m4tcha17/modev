# ADR-007: Boundary-Region Tolerance Band

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** §11, §15 item 7

## Default

**±0.5 percentage points** around each threshold (6.0%/6.1% and 13.9%/14.0%). "Near-boundary" = moisture reading within 0.5pp of either threshold; everything else is "mid-range." Both buckets get separate metrics in evaluation (`evaluation.py`).

## Rationale

The Brown-Duvel meter's documented reading-to-reading variance matters most exactly at the two thresholds — a sample truly at 6.05% could plausibly read anywhere from ~5.5% to ~6.5% depending on measurement noise. A ±0.5pp band captures that variance window without being so wide it swallows most of Class 2.

## What would change it

If the actual documented Brown-Duvel variance figure is smaller or larger than 0.5pp, tighten or widen the band to match it directly rather than using this placeholder.

## Owner

Thesis team; ideally set from the Brown-Duvel meter's actual documented variance spec rather than this estimate.
