"""Background-only lighting check (REQ-4.1-8, ADR-013 item 2): can the grade
be predicted from the paper alone? Runs on the raw photos and after
grey-paper normalization, and reports both against chance and the
leak threshold (config evaluation.lighting_leak_accuracy).

Usage:
  uv run python scripts/lighting_check.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

from copra_grading.dataset import assert_clean, balance_cubes, drop_excluded, load_dataset, load_image
from copra_grading.lighting_check import grade_accuracy_from_paper, paper_features


def _photo(csv_path: Path, row: dict, reference_grey: float) -> tuple[dict, dict]:
    image = load_image(csv_path, row["path"])
    keys = {c: row[c] for c in ("id", "batch_id", "copra_class")}
    return ({**keys, **paper_features(image)},
            {**keys, **paper_features(image, reference_grey)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    csv_path = Path(config["data"]["csv_path"])
    df = load_dataset(csv_path)
    assert_clean(df)
    exclusions = config["data"].get("mask_exclusions")
    df, _ = drop_excluded(df, Path(exclusions) if exclusions else None)
    df, _ = balance_cubes(df, config["data"]["min_grade_ratio"], config["data"]["balance_seed"])
    reference = config["preprocessing"]["lighting_normalization"]["reference_grey"]

    pairs = Parallel(n_jobs=args.n_jobs)(
        delayed(_photo)(csv_path, row, reference) for row in df.to_dict(orient="records")
    )
    threshold = config["evaluation"]["lighting_leak_accuracy"]
    report = {"leak_threshold": threshold}
    for stage, rows in (("before_normalization", [p[0] for p in pairs]),
                        ("after_normalization", [p[1] for p in pairs])):
        table = pd.DataFrame(rows)
        result = grade_accuracy_from_paper(table, config)
        result["leak"] = result["mean_accuracy"] > threshold
        result["paper_lab_mean_by_class"] = (
            table.groupby("copra_class")[["paper_l_mean", "paper_a_mean", "paper_b_mean"]]
            .mean().round(1).to_dict(orient="index")
        )
        report[stage] = result
        print(f"{stage}: accuracy {result['mean_accuracy']:.3f} "
              f"(chance {result['chance']:.3f}, leak above {threshold}) -> "
              f"{'LEAK' if result['leak'] else 'ok'}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lighting_check.json").write_text(json.dumps(report, indent=2))
    print(f"-> {args.out / 'lighting_check.json'}")


if __name__ == "__main__":
    main()
