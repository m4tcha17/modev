# ADR-006: GLCM Distance Set

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** `process.md` Step 3a ("a few pixel distances")

## Default

Distances: **{1, 2, 3} pixels**. Angles: **{0°, 45°, 90°, 135°}** (all four, per process.md). Each distance/angle combination kept as a **separate feature** (not averaged across distances) — a defensible, documented choice per the spec's own framing; this preserves more information for the tree-based models to split on than averaging would.

## Rationale

1-3px captures fine-grained surface texture at the fixed 10 cm shooting distance after resize, without the feature count exploding. Keeping distances separate (vs. averaged) costs little for tree ensembles, which handle correlated/high-dimensional features natively.

## What would change it

If TreeSHAP (Step 8) shows negligible marginal contribution from farther distances (e.g. distance=3 barely used), the set could shrink; if texture signal looks distance-sensitive in ways 1-3px doesn't capture, extend the range.

## Owner

Thesis team; can be empirically revisited once TreeSHAP results are available.
