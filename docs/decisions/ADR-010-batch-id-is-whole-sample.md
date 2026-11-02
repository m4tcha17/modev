# ADR-010: `batch_id` Is the Whole-Sample ID

**Status:** adopted 2026-10-10, amended 2026-10-31 (batch_id = cube; see end)
**Spec reference:** `model-development-instructions.md` §3, §7a, §8b; acceptance P4; hard ordering constraint in `CLAUDE.md`

## Context

`CLAUDE.md` and `configs/default.yaml` warned that several `batch_id`s might come from one whole copra sample, in which case grouping by `batch_id` would leak one sample across folds, and asked for a whole-sample ID column.

## Decision

Confirmed by the thesis team: one `batch_id` = one whole copra sample. Each sample is photographed four times, one photo per side of the inside meat, and those four photos share the `batch_id`. No sample is split across batches.

`splitting.group_column` stays `batch_id`. No new column is needed.

Checked on the 2026-10-05 export: 94 batches, every batch has exactly 4 photos and exactly one `copra_class` (A: 50 batches, B: 44). Rechecked on the 2026-10-10 export: 144 batches, same result (A: 50, B: 44, D: 50).

## What would change it

If data collection ever photographs one sample in more than one session under different `batch_id`s (e.g. re-shooting a sample later), this stops holding. Add a whole-sample ID column then and point `splitting.group_column` at it.

## Amendment 2026-10-31: `batch_id` is the cube, grouping stays on it

The spec (`model-development-instructions.md` §3, §7a) defines `batch_id` as the **Cube ID**; one copra is cut into 1-3 cubes. Which cubes came from the same copra was not recorded, so the spec now groups by cube (`batch_id`) at both nested-CV levels (outer 5, inner 3). No `copra_id` column exists or will be added.

Limitation to report (acceptance §6): cubes cut from one copra look alike and may fall in different folds, so cross-validated scores may be somewhat optimistic. The earlier Decision above (one `batch_id` = one whole sample) is superseded.
