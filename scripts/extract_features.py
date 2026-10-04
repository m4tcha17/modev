"""Run Steps 1-3 on a dataset export: write the feature table and a mask
preview sheet for checking masks by eye.

The table has one row per photo per rotation/flip (8 per photo; column
`augment`, "rot0" = original) so training can pick augmented copies without
re-extracting. Use --no-augment for originals only.

Usage:
  uv run python scripts/extract_features.py                     # paths from configs/default.yaml
  uv run python scripts/extract_features.py --csv path/to/copra-dataset.csv --out data/processed
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from copra_grading.dataset import assert_clean, load_dataset, load_image
from copra_grading.augmentation import ORIGINAL, TRANSFORMS
from copra_grading.features.table import build_feature_table, feature_columns, originals
from copra_grading.preprocessing.pipeline import compute_mask

TILE = (320, 240)


def write_mask_preview(df, csv_path: Path, config: dict, out_path: Path, n: int = 24) -> None:
    """Grid of photos with the mask outline drawn in red, evenly sampled."""
    picks = np.linspace(0, len(df) - 1, min(n, len(df))).astype(int)
    tiles = []
    for i in picks:
        row = df.iloc[i]
        image = load_image(csv_path, row["path"])
        mask = compute_mask(image, config)
        tile = cv2.resize(image, TILE, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask.astype(np.uint8), TILE, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(tile, contours, -1, (255, 0, 0), 2)
        cv2.putText(tile, f"{i} {row['copra_class']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[j : j + 4]) for j in range(0, len(tiles), 4)])
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-augment", action="store_true", help="originals only, no rotated/flipped copies")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    csv_path = args.csv or Path(config["data"]["csv_path"])
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    assert_clean(df)
    print(f"{len(df)} photos, {df['batch_id'].nunique()} batches, classes: "
          f"{df['copra_class'].value_counts().sort_index().to_dict()}")

    start = time.time()
    transforms = (ORIGINAL,) if args.no_augment else TRANSFORMS
    table = build_feature_table(df, csv_path, config, transforms=transforms, n_jobs=args.n_jobs)
    table_path = args.out / "features.csv"
    table.to_csv(table_path, index=False)
    print(f"{len(feature_columns(table))} features x {len(table)} rows "
          f"({len(df)} photos x {len(transforms)} orientations) -> {table_path} ({time.time() - start:.0f}s)")

    frac = originals(table)["mask_fraction"]
    print(f"mask_fraction min/median/max: {frac.min():.3f} / {frac.median():.3f} / {frac.max():.3f}")

    preview_path = args.out / "mask_preview.jpg"
    write_mask_preview(df, csv_path, config, preview_path)
    print(f"mask preview -> {preview_path}")


if __name__ == "__main__":
    main()
