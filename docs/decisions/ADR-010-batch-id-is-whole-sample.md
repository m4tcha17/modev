# ADR-010: `batch_id` Is the Whole-Sample ID

**Status:** adopted 2026-10-10
**Spec reference:** `process.md` Step 4b (StratifiedGroupKFold, grouped by whole copra sample); hard ordering constraint in `CLAUDE.md`

## Context

`CLAUDE.md` and `configs/default.yaml` warned that several `batch_id`s might come from one whole copra sample, in which case grouping by `batch_id` would leak one sample across folds, and asked for a whole-sample ID column.

## Decision

Confirmed by the thesis team: one `batch_id` = one whole copra sample. Each sample is photographed four times, one photo per side of the inside meat, and those four photos share the `batch_id`. No sample is split across batches.

`splitting.group_column` stays `batch_id`. No new column is needed.

Checked on the 2026-10-05 export: 94 batches, every batch has exactly 4 photos and exactly one `copra_class` (A: 50 batches, B: 44).

## What would change it

If data collection ever photographs one sample in more than one session under different `batch_id`s (e.g. re-shooting a sample later), this stops holding. Add a whole-sample ID column then and point `splitting.group_column` at it.
