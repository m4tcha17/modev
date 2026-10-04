"""Run background masking on every photo of an export and save the results
for checking by eye.

Writes, for each photo `<id>`:
  <out>/masked/<id>.jpg    copra only, cropped, background black
  <out>/overlays/<id>.jpg  original photo (downscaled) with the mask outline in red
and <out>/mask_summary.csv  (id, batch_id, copra_class, mask_fraction).

Uses the masking method in the config (`preprocessing.masking_method`,
default background_grabcut). Output is for viewing only - the pipeline
recomputes masks itself.

Usage:
  uv run python scripts/mask_photos.py                         # all photos -> dataset/masked_data
  uv run python scripts/mask_photos.py --limit 10              # first 10 photos only
  uv run python scripts/mask_photos.py --csv path/to/copra-dataset.csv --out path/to/folder
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from copra_grading.dataset import load_dataset, load_image
from copra_grading.preprocessing.otsu import apply_mask
from copra_grading.preprocessing.pipeline import compute_mask

OVERLAY_WIDTH = 1020


def _save_rgb(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def mask_one(csv_path: Path, row: dict, config: dict, out: Path) -> dict:
    image = load_image(csv_path, row["path"])
    mask = compute_mask(image, config)

    _save_rgb(out / "masked" / f"{row['id']}.jpg", apply_mask(image, mask))

    h, w = image.shape[:2]
    size = (OVERLAY_WIDTH, round(h * OVERLAY_WIDTH / w))
    overlay = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 3)
    _save_rgb(out / "overlays" / f"{row['id']}.jpg", overlay)

    return {
        "id": row["id"],
        "batch_id": row["batch_id"],
        "copra_class": row["copra_class"],
        "mask_fraction": round(float(mask.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--csv", type=Path, default=None, help="default: data.csv_path in the config")
    parser.add_argument("--out", type=Path, default=Path("dataset/masked_data"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N photos")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all CPU cores)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    csv_path = args.csv or Path(config["data"]["csv_path"])
    df = load_dataset(csv_path)
    if args.limit:
        df = df.head(args.limit)

    for sub in ("masked", "overlays"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    print(f"masking {len(df)} photos from {csv_path} "
          f"(method: {config['preprocessing']['masking_method']}) ...")
    start = time.time()
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(mask_one)(csv_path, record, config, args.out)
        for record in df.to_dict(orient="records")
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "mask_summary.csv", index=False)

    frac = summary["mask_fraction"]
    print(f"done in {time.time() - start:.0f}s -> {args.out}/")
    print(f"  masked/    {len(summary)} images (copra only, black background)")
    print(f"  overlays/  {len(summary)} images (red outline on the original)")
    print(f"  mask_fraction min/median/max: {frac.min():.3f} / {frac.median():.3f} / {frac.max():.3f}")
    odd = summary[(frac < 0.05) | (frac > 0.5)]
    if len(odd):
        print(f"  check these (mask unusually small or large): {', '.join(odd['id'])}")


if __name__ == "__main__":
    main()
