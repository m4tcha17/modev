# ADR-008: Masking Method — Background Model + GrabCut Instead of Otsu

**Status:** adopted 2026-10-04, pending thesis-team confirmation
**Spec reference:** `process.md` Step 2 ("If Otsu turns out unreliable, fallbacks are adaptive thresholding, an HSV-based mask, or GrabCut. Only switch if Otsu actually fails.")

## Otsu failed

On the 2026-10-04 export (136 photos), Otsu's mask covered a median 77% of the frame. Copra is two-toned: white meat (LAB L ≈ 200) and dark brown skin (L ≈ 30). The mid-grey background (L ≈ 100–170, drifting across the frame with the lighting) lies between them, so one global threshold groups the background with the meat. Features were computed mostly on background.

## Decision

`preprocessing/background.py`, `masking_method: background_grabcut`:

1. Fit a quadratic surface per LAB channel to the image border (assumed background), which captures the lighting gradient.
2. Copra seeds = pixels more than `background_threshold_k` (4) robust SDs from that surface.
3. GrabCut refines the edge from those seeds. Keep the largest piece, fill holes.

Result on the same 136 photos: mask covers 13–37% of the frame, checked by eye on a 24-photo preview sheet. A few masks pick up a thin sliver of cast shadow at one edge.

Also adopted: GLCM and Canny are measured inside the copra only (`features.exclude_boundary: true`). GLCM counts only copra–copra pixel pairs; Canny ignores edges within `boundary_margin_px` (3) of the outline. Before this, the outline cliff inflated GLCM contrast and Canny counted the outline as cracks.

## Assumptions it relies on

One copra piece per photo, not touching the frame edge, on the uniform grey background. A photo that breaks these (piece cut off at the edge, two pieces, different background) will mask badly. `tests/test_real_dataset.py` checks size and border on sampled real photos.

## What would change it

A new background or a different shooting setup: re-run `scripts/extract_features.py` and check `mask_preview.jpg`.
