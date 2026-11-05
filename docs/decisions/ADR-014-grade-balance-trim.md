# ADR-014: Trim Over-Represented Grades to the 90% Balance Rule

**Status:** accepted 2026-11-05 (student decision).
**Spec reference:** thesis manuscript §4.1.3 and Table 1: the smallest grade's cube count is held at no less than 90% of the largest grade's.

## Context

The full A–F export (2026-11-05, 2,652 photos) came in uneven: A 151 cubes, B 100, C 100, D 99, E 101, F 111. D is at 66% of A, well short of the thesis rule. Class weighting and class-targeted augmentation would absorb a gap this size, but the thesis states the dataset itself meets the 90% rule, and the student chose to make the data match the thesis rather than rewrite the rule.

## Decision

`dataset.balance_cubes` runs in `extract_features.py` (and `lighting_check.py`) right after mask exclusions, before any feature is extracted:

- cap = the largest n with smallest-grade cubes ≥ `data.min_grade_ratio` × n. With D at 99 and ratio 0.9, cap = 110.
- Every grade above the cap loses whole cubes (batches) down to the cap, never single photos, so kept cubes keep all four faces.
- Which cubes go is a seeded random draw (`data.balance_seed`) over sorted batch ids: reproducible, and not biased toward a collection date or session (later cubes may share a lighting session).
- The dropped batch ids are written to `data/processed/balance_dropped.csv` for the record.
- The raw export is not edited. Changing the ratio or the seed regenerates the trim.

## Consequences

- On the 2026-11-05 export: A loses 41 cubes (164 photos), F loses its excess. 2,484 photos remain before outlier removal.
- The cap uses cube counts at export, before the outlier drop. Outlier removal drops photos, not cubes, so per-grade photo counts can still drift apart after it; class weighting and class-targeted augmentation cover that, as before.
- Class weighting stays on in all four models. It is now correcting a small residual gap instead of a 1.5× one.
- Discarded cubes are real data. If D or another small grade gains cubes later, the cap rises and fewer cubes are dropped.
