# ADR-004: Image File Delivery Format Alongside CSV Export

**Status:** proposed-default, pending data-collection-team confirmation
**Spec reference:** §3, §15 item 4

## Default

No assumed layout. Data loading code takes a **configurable image root path** (`configs/default.yaml: data.image_root`) and resolves images per row via `Sample_ID`/`Angle_ID`, tolerant of flat-folder, zip, or cloud-folder delivery as long as the root path resolves to something file-path-addressable (a mounted/extracted directory).

## Rationale

Exact delivery format (flat folder vs. zip vs. cloud folder) isn't finalized by the data-collection side as of this scaffold. Hardcoding a layout would require rework the moment the real export format is confirmed. This is already a hard requirement in §3, not just a preference — this ADR records it as the pending-confirmation item it is.

## What would change it

Once the Jotter/Supabase export format is finalized, confirm whether a zip needs extraction before pointing `image_root` at it, or whether cloud-folder delivery needs a sync/download step before the pipeline runs.

## Owner

Data-collection team (Jotter/Supabase export format decision).
