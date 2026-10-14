# ADR-011: Class D Lighting and Outlier Loss on the 2026-10-10 Export

**Status:** adopted 2026-10-11
**Spec reference:** `CLAUDE.md` (outliers dropped, never relaxed, "collect more data" if too much is lost); `process.md` Steps 4a–4b

## Context

The 2026-10-10 export (`dataset/copra_dataset/copra-dataset.csv`) has 576 photos in 144 batches: class `A` 200 photos (50 batches), `B` 176 (44), `D` 200 (50). Classes `C`, `E` and `F` are still being collected.

Two problems were found on it:

1. **Outlier loss.** With the fixed thresholds (|z| > 3 or outside 1.5×IQR, original rows only), 182 of 576 photos (32%) are dropped: A 46, B 60, D 76. Kept: A 154 photos (47 batches), B 116 (41), D 124 (45). The 12 `glcm_*_energy` columns flag most of these (40–47 photos each, largely the same photos); `edge_contour_max_length` flags 40. 74 photos are dropped by a single feature.
2. **Class D was photographed under different conditions.** The class D photos have a visibly lighter background than A and B, and the mask leaks into shadow on several of them (`data/processed/mask_preview.jpg`, tiles 400, 475, 500, 550). Lighting changes the copra's own color as well as the background, so the HSV/LAB features for D partly measure the photo setup rather than moisture.

## Decision

Confirmed by the thesis team: the class D photos cannot be retaken. More data (classes C, E, F) is still coming.

- Outlier thresholds stay as they are (`CLAUDE.md`). The loss is addressed by the incoming data, not by relaxing anything. Outlier statistics are recomputed on each new export, so the drop count will change.
- The class D lighting difference is a stated thesis limitation. Scores involving class D (its per-class F1 and its confusion-matrix cells) may be inflated by it and must be reported with that caveat.
- Model development can start on A/B/D. Its results are interim until the remaining classes arrive.

## What would change it

- **Lighting:** if C/E/F turn out to be shot under the D setup, the confound becomes setup-vs-setup rather than D-vs-rest; recheck the mask preview and per-class background brightness on each new export.
- **Outlier loss:** if the drop rate stays near a third once all six classes are in, revisit which features cause it (the redundant `glcm_*_energy` columns) in a new ADR. The thresholds themselves stay fixed.
