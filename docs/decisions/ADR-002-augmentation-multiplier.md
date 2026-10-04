# ADR-002: Per-Class Augmentation Multiplier

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** `process.md` Step 4c

## Fixed by process.md

Training folds only, after the StratifiedGroupKFold split. Rotations and flips only — no brightness, contrast, or color changes, no SMOTE. Smaller classes get more augmented copies than larger ones. Augmented images go through masking + feature extraction like originals.

## Default (open part)

Multipliers computed per training fold from class counts (`augmentation.class_multipliers`): the largest class gets the fewest copies, smaller classes proportionally more, capped at `augmentation.max_multiplier` (default **4**) to limit near-duplicates of the smallest class.

## What would change it

Real per-class counts across A–F. The 2026-10-04 export is all class `A`, so the cap can't be tuned yet.

## Owner

Thesis team (data/methodology decision).
