# ADR-009: Color Features — Mean and Std Only, No Percentiles

**Status:** adopted 2026-10-05
**Spec reference:** `process.md` Step 3b (HSV/LAB color features); outlier rule in `CLAUDE.md` (drop, never impute, never relax thresholds)

## Context

Commit `01dbddb` added median, p10 and p90 next to mean and std for every HSV/LAB channel (12 → 30 color features). Each photo row with any outlier value is dropped, so every extra feature is another chance for a photo to be thrown away.

Measured on the 2026-10-05 export (376 photos, classes A and B), originals only, |z| > 3 or outside 1.5×IQR:

| Feature set | Photos dropped |
|---|---|
| Before `01dbddb` | 112 |
| `01dbddb` as committed | 176 (46.8%) |
| Hue percentiles removed | 157 (41.8%) |
| All color percentiles removed (this ADR) | 123 (32.7%) |

## Why the percentiles went

- **Hue:** linear percentiles on a circular value are meaningless (179 and 0 are neighbours), and near-white copra meat has noise-level hue spread around the whole wheel.
- **Other channels:** OpenCV channels are uint8, so a median/percentile is a whole number. On real photos they bunch tightly (`lab_a_median`: 175 of 376 photos at exactly 128, nearly all within 127–130), so the IQR fence is a couple of units wide and ordinary photos at 126 or 132 get flagged. They were the single largest source of extra drops, not real outliers.

Mean and std over thousands of copra pixels are continuous and did not have this problem. Hue keeps the circular mean/std from `01dbddb`.

## Kept from 01dbddb

The new edge contour stats (`edge_contour_density`, `edge_contour_mean_length`, `edge_contour_max_length`) stay; they add 11 drops (146 vs 157 with them). Revisit if the model comparison shows they add nothing.

## What would change it

More data with a wider moisture range (classes C–F) could spread the percentile values out. Re-measure drop counts before re-adding them. The thresholds are not the thing to change.
