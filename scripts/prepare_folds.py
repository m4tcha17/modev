"""Run Step 4 on a feature table from extract_features.py: drop outlier
photos (4a), split into StratifiedGroupKFold folds (4b), pick each training
fold's augmented copies (4c).

Writes the cleaned table and a fold assignment per photo (the fold it is
validated in). Training code rebuilds the same folds with folds.make_folds.

Usage:
  uv run python scripts/prepare_folds.py                  # data/processed/features.csv
  uv run python scripts/prepare_folds.py --features path/to/features.csv --out data/processed
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from copra_grading.cleaning import outlier_value_mask, remove_outlier_photos
from copra_grading.features.table import ID_COLUMNS, originals
from copra_grading.folds import make_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    args.out.mkdir(parents=True, exist_ok=True)

    # 4a: outliers, judged on original rows only
    clean, flags = remove_outlier_photos(table, config)
    n_photos = len(flags)
    print(f"outliers: dropped {int(flags.sum())} of {n_photos} photos "
          f"({flags.mean():.1%}), {clean['id'].nunique()} kept")
    cfg = config["cleaning"]
    hits = outlier_value_mask(originals(table), cfg["zscore_threshold"], cfg["iqr_multiplier"]).sum()
    top = hits[hits > 0].sort_values(ascending=False).head(10)
    if not top.empty:
        print("most-flagged features (photos):")
        for name, n in top.items():
            print(f"  {name}: {n}")
    before = originals(table)["copra_class"].value_counts().sort_index()
    after = originals(clean)["copra_class"].value_counts().reindex(before.index, fill_value=0)
    print("photos per class, before -> after: "
          + ", ".join(f"{c} {before[c]}->{after[c]}" for c in before.index))

    clean_path = args.out / "features_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"cleaned table -> {clean_path}")

    # 4b + 4c: split originals, then pick augmented copies per training fold
    folds = make_folds(clean, config)
    group = config["splitting"]["group_column"]
    assignments = []
    for fold in folds:
        val = fold.val
        print(f"fold {fold.index}: train {len(fold.train)} rows "
              f"({fold.train['id'].nunique()} photos + augmented), "
              f"val {len(val)} photos / {val[group].nunique()} groups, "
              f"val classes {val['copra_class'].value_counts().sort_index().to_dict()}, "
              f"multipliers {fold.multipliers}")
        assignments.append(val[list(dict.fromkeys([*ID_COLUMNS, group]))].assign(fold=fold.index))

    folds_path = args.out / "folds.csv"
    pd.concat(assignments).sort_values(["fold", "id"]).to_csv(folds_path, index=False)
    print(f"fold assignments -> {folds_path}")


if __name__ == "__main__":
    main()
