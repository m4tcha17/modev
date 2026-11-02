"""Background-only check (T5): can the grade be predicted from the grey paper
alone? Run before and after lighting normalization (both modes: gain, flat_field).

Per photo: mask the copra, take paper pixels outside a dilated mask, compute
RGB/LAB mean+SD. Then grouped 5-fold CV (same split rules as training) of a
Logistic Regression and a Random Forest. Also prints the dataset's median
paper color, the suggested `preprocessing.reference_grey`.

GrabCut runs once per photo - use --n-jobs to cap CPU.

Usage:
  uv run python scripts/background_check.py --n-jobs 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from copra_grading.background_check import classify_from_paper, paper_features, paper_pixels
from copra_grading.dataset import load_dataset, load_image
import cv2

from copra_grading.preprocessing.lighting import normalize, paper_color
from copra_grading.preprocessing.pipeline import compute_mask


MODES = ("gain", "flat_field")


def _photo(csv_path: Path, row: dict, config: dict, reference) -> dict:
    # one thread per joblib worker (OpenCV + BLAS), so --n-jobs is the real core count
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        return _photo_stats(csv_path, row, config, reference)


def _photo_stats(csv_path: Path, row: dict, config: dict, reference) -> dict:
    image = load_image(csv_path, row["path"])
    mask = compute_mask(image, config)
    paper = paper_color(image, mask)
    out = {"id": row["id"], "batch_id": row["batch_id"], "copra_class": row["copra_class"],
           **{f"measured_paper_{c}": float(v) for c, v in zip("rgb", paper)}}
    before = paper_features(paper_pixels(image, mask))
    out.update({f"before_{k}": v for k, v in before.items()})
    for mode in MODES:
        after = paper_features(paper_pixels(normalize(image, mask, reference, mode), mask))
        out.update({f"{mode}_{k}": v for k, v in after.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    csv_path = Path(config["data"]["csv_path"])
    reference = config["preprocessing"]["reference_grey"]
    split = config["splitting"]
    margin = config["targets"]["background_check_margin"]
    df = load_dataset(csv_path)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(_photo)(csv_path, r, config, reference) for r in df.to_dict(orient="records")
    )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "background_check_photos.csv", index=False)

    measured = table[[f"measured_paper_{c}" for c in "rgb"]]
    print(f"{len(table)} photos. measured paper RGB under copra, median "
          f"{np.round(measured.median().to_numpy(), 1).tolist()} (suggested reference_grey), "
          f"configured reference_grey {reference}")
    print("measured paper L-ish brightness (mean RGB) per class:")
    bright = measured.mean(axis=1).groupby(table["copra_class"]).describe()[["mean", "std", "min", "max"]]
    print(bright.round(1).to_string())

    n_classes = table["copra_class"].nunique()
    chance = 1 / n_classes
    limit = chance + margin
    print(f"\nchance = 1/{n_classes} = {chance:.3f}; T5 passes if accuracy <= {limit:.3f}")
    results = []
    for stage in ("before", *MODES):
        cols = [c for c in table.columns if c.startswith(f"{stage}_")]
        scores = classify_from_paper(table, cols, split["n_splits"], split["group_column"],
                                     split["random_state"]).assign(stage=stage)
        results.append(scores)
        for name, s in scores.groupby("model"):
            acc = s["accuracy"]
            verdict = "PASS" if acc.mean() <= limit else "FAIL"
            print(f"  {stage:10s} {name:20s} accuracy {acc.mean():.3f} ± {acc.std():.3f}  "
                  f"macro F1 {s['macro_f1'].mean():.3f}  {verdict}")
    pd.concat(results).to_csv(args.out / "background_check_scores.csv", index=False)
    print(f"\nper-photo stats -> {args.out / 'background_check_photos.csv'}")


if __name__ == "__main__":
    main()
