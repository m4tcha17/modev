# ADR-002: Per-Class Augmentation Multiplier

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** §7b, §15 item 2

## Default

Configurable dict, e.g. `{1: 3, 2: 1, 3: 3}` (illustrative starting point — real multipliers depend on actual collected class counts). Rule for picking the starting values: choose multipliers that roughly balance *effective* post-augmentation class counts, given natural skew toward Class 2 (§3).

Geometric transforms only: rotations and flips. No photometric augmentation, no SMOTE (see `../../CLAUDE.md`).

## Rationale

Real collected class counts aren't known yet at scaffold time, so exact multipliers can't be fixed. The dict interface (`augmentation/geometric.py`) lets the future session plug in real per-class counts and compute multipliers once the dataset export exists, without changing the augmentation code itself.

## What would change it

Once real per-class sample counts are known, compute multipliers that bring Class 1 and Class 3 effective counts close to Class 2's, while checking augmented-copy volume doesn't produce excessive near-duplicates for the smallest class.

## Owner

Thesis team (data/methodology decision).
