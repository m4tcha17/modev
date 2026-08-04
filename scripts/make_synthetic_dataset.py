"""Generate a schema-matching synthetic dataset for pipeline development
before real field data lands. Spec §15 closing note.

Produces: a CSV with Sample_ID, Angle_ID, timestamp, moisture_reading (one row
per angle image, six per sample) plus placeholder images under a configurable
output image root - same shape the pipeline expects from a real export.

Usage: uv run python scripts/make_synthetic_dataset.py --n-samples 200 --out-dir data/raw
"""

import argparse
from pathlib import Path


def generate_synthetic_dataset(n_samples: int, out_dir: Path) -> None:
    """Write export.csv + placeholder angle images to out_dir, class-balanced
    across the three moisture brackets per §1's thresholds.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    generate_synthetic_dataset(args.n_samples, args.out_dir)


if __name__ == "__main__":
    main()
