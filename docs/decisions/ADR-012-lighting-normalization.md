# ADR-012: Lighting Normalization and the Background-Only Check

**Related:** ADR-011 (Grade D lighting, team-confirmed 2026-10-11).
**Status:** implemented 2026-10-31; T5 fails on the current export (see Result)
**Spec reference:** `model-development-instructions.md` §4 (lighting normalization), §11 (background-only check); acceptance L2 (was `goal.md` T5)

## Decision

After masking, before features, rescale each photo so its grey paper matches `preprocessing.reference_grey` (`preprocessing/lighting.py`). The paper color comes from a quadratic surface fitted to the image border band (same robust fit as masking), averaged under the copra. Two modes, `preprocessing.lighting_mode`:

- `gain` (default, the spec's wording): one gain per RGB channel.
- `flat_field`: divide each pixel by the fitted paper surface at that point, which also removes the lighting gradient.

`reference_grey` = [117, 125, 124], the median measured paper color of the 2026-10-31 export (A/B/D, 576 photos), so a typical photo gets a gain near 1.

`scripts/background_check.py` runs T5: paper-only pixels (outside a dilated copra mask) → RGB/LAB mean+SD → grouped 5-fold CV, Logistic Regression and Random Forest, before and after both modes. Acceptance L2: accuracy near chance (~17% for six grades); **above 25% is investigated as lighting leakage**, so pass = accuracy ≤ 25% after normalization. Report before and after. The code currently uses chance + `targets.background_check_margin` (0.08), which equals ~0.25 only with six grades; with fewer grades in an export it is a different limit. Align it with the absolute 25% mark (gap listed in `docs/architecture.md`).

## Result on the 2026-10-31 export

Chance 0.333 (3 grades), code's pass limit 0.413 (chance + 0.08). Indicative only: L2 is defined for six grades.

| Stage | LR accuracy | RF accuracy |
|---|---|---|
| Before normalization | 0.878 | 0.913 |
| `gain` | 0.764 | 0.783 |
| `flat_field` | 0.754 | 0.826 |

- Grade D paper is ~50% brighter before normalization (mean RGB ~170 vs ~113 for A/B): D was not shot under the same lighting, not just after some dimming.
- After normalization the paper means match, but paper texture/noise SD still differs (A/B ~10.5, D ~5.7 after flat-field). Per-photo rescaling can't remove that.
- A vs B alone (chance 0.5): 0.83-0.87 before, 0.75-0.79 after flat-field. Each grade's capture session has its own fingerprint.

Cause: grades were photographed in sequence, so capture session and grade are fully confounded in this export. Normalization reduces the signal but does not remove it. Per acceptance L2, M1-M4 can't be trusted on this export until the leak is fixed, and this is reported, not tuned around.

Caveat: paper pixels near the copra may still include some copra-dependent shadow. A border-band-only variant would rule that out; the D brightness gap alone already rules out a copra-only explanation.

## Decision 2026-11-02: accepted as a known limitation

Re-capturing with grades mixed within each session is not possible, so the team proceeds with the data as collected. Rules that follow from this:

- Lighting normalization stays on, and the background-only check is still run before and after it on every export and reported (acceptance L2, report item 6).
- If L2 fails, the results summary says so next to M1–M4, in plain words: the model may partly be learning the capture session's lighting instead of the copra, so its scores are likely optimistic.
- Never tune, change features, or drop data to make L2 pass. Report the actual numbers.

## What would change it

Re-capture with grades mixed within each session (each session photographs some cubes of every grade), so a session can't stand in for a grade. Not possible for this study as of 2026-11-02.
