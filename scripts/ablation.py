"""Feature-family ablation (thesis section 4.10, ADR-013 item 3): rerun the
same nested CV with only the texture (GLCM), color (HSV/LAB) or edge (Canny)
features, and compare each against the combined set from train_models.py.

Pass mark: the combined set scores no lower on mean outer-fold Macro F1 than
the best single family. Same outer folds, seeds and trial budget as
train_models.py, so the numbers are directly comparable.

Usage:
  uv run python scripts/ablation.py                          # selected model only
  uv run python scripts/ablation.py --models lightgbm logistic_regression
  uv run python scripts/ablation.py --n-trials 3             # smoke run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from copra_grading.explainability import FAMILIES, feature_family
from copra_grading.features.table import ID_COLUMNS, META_COLUMNS, feature_columns
from copra_grading.models import MODEL_NAMES
from copra_grading.selection import cross_validate_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_clean.csv"))
    parser.add_argument("--results", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    n_trials = args.n_trials or config["tuning"]["n_trials"]
    models = tuple(args.models or [json.loads((args.results / "selection.json").read_text())["selected"]])
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    combined = pd.read_csv(args.results / "cv_scores.csv")

    rows = []
    for family in FAMILIES:
        columns = [c for c in feature_columns(table) if feature_family(c) == family]
        subset = table[[c for c in table.columns if c in META_COLUMNS] + columns]
        print(f"--- {family}: {len(columns)} features", flush=True)
        result = cross_validate_models(subset, config, models, n_trials)
        for name, scores in result.scores.items():
            rows += [{"model": name, "features": family, "fold": k, "macro_f1": s} for k, s in enumerate(scores)]
    for name in models:
        kept = combined[combined["model"] == name]
        if kept.empty:
            raise SystemExit(f"{name} has no combined-feature scores in {args.results / 'cv_scores.csv'}")
        rows += [{"model": name, "features": "combined", "fold": r.fold, "macro_f1": r.macro_f1}
                 for r in kept.itertuples()]

    scores = pd.DataFrame(rows)
    scores.to_csv(args.results / "ablation_scores.csv", index=False)
    summary = scores.groupby(["model", "features"])["macro_f1"].agg(["mean", "std"]).round(4)
    report = {}
    for name in models:
        means = summary.loc[name, "mean"]
        best_family = means.drop("combined").idxmax()
        report[name] = {
            "mean_macro_f1": means.to_dict(),
            "best_single_family": best_family,
            "combined_not_lower": bool(means["combined"] >= means[best_family]),
        }
    (args.results / "ablation.json").write_text(json.dumps(report, indent=2))
    print(summary.to_string())
    for name, r in report.items():
        print(f"{name}: combined {'>=' if r['combined_not_lower'] else '<'} best single family "
              f"({r['best_single_family']}) -> {'PASS' if r['combined_not_lower'] else 'MISS'}")


if __name__ == "__main__":
    main()
