# ADR-004: Image File Delivery Format Alongside CSV Export

**Status:** resolved by `process.md` §1 and the 2026-10-04 export

## Decision

`copra-dataset.csv` and a `photos/` folder sit in the same directory. Each photo is `photos/<id>.<ext>` and the CSV's `path` column holds that relative path. `dataset.load_image` resolves `path` against the CSV's directory; `configs/default.yaml: data.csv_path` is the only setting. A delivered zip must be extracted first.

## Superseded

The earlier "no assumed layout, resolve via `Sample_ID`/`Angle_ID`" default.
