# ADR-002: Per-Class Augmentation Multiplier

**Status:** proposed-default, pending thesis-team confirmation (open item §15.1)
**Spec reference:** `model-development-instructions.md` §7b, §8b

## Fixed by the spec

After every split, never before: inner training folds during tuning, the whole outer training portion when refitting (§8b). Rotations and flips only — no brightness, contrast, or color changes, no SMOTE. Smaller classes get more augmented copies than larger ones. Augmented images go through masking + feature extraction like originals.

## Default (open part)

Multipliers computed per training fold from class counts (`augmentation.class_multipliers`): the largest class gets the fewest copies, smaller classes proportionally more, capped at `augmentation.max_multiplier` (default **4**) to limit near-duplicates of the smallest class.

## What would change it

Real per-grade counts across A–F. The current export has only A, B, D (50/44/50 cubes), so the cap can't be tuned yet.

## Owner

Thesis team (data/methodology decision).
