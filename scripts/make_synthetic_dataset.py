"""Generate a schema-matching synthetic dataset for pipeline development.

Produces: copra-dataset.csv with id, batch_id, copra_class, path (one row per
photo, four per batch_id) plus placeholder images under <out-dir>/photos/ -
same layout as a real export (see process.md §1).

Usage: uv run python scripts/make_synthetic_dataset.py --n-batches 60 --out-dir data/raw
"""

import argparse
from pathlib import Path


def generate_synthetic_dataset(n_batches: int, out_dir: Path) -> None:
    """Write copra-dataset.csv + photos/<id>.jpg to out_dir, batches spread
    across classes A-F, four photos per batch sharing one copra_class.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batches", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    generate_synthetic_dataset(args.n_batches, args.out_dir)


if __name__ == "__main__":
    main()
