"""Run Steps 5-7 on the cleaned feature table from prepare_folds.py:
nested-CV tuning and training of all four models, selection of the best
ensemble, and per-photo metrics on the held-out folds.

Usage:
  uv run python scripts/train_models.py                        # all models, config trials
  uv run python scripts/train_models.py --n-trials 3 --models random_forest   # smoke run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from copra_grading.evaluation import compute_metrics
from copra_grading.features.table import ID_COLUMNS
from copra_grading.models import MODEL_NAMES
from copra_grading.selection import ENSEMBLE_CANDIDATES, cross_validate_models, select_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    n_trials = args.n_trials or config["tuning"]["n_trials"]
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    args.out.mkdir(parents=True, exist_ok=True)

    result = cross_validate_models(table, config, tuple(args.models), n_trials)

    scores = pd.DataFrame(
        [(m, k, s) for m, fold_scores in result.scores.items() for k, s in enumerate(fold_scores)],
        columns=["model", "fold", "macro_f1"],
    )
    scores.to_csv(args.out / "cv_scores.csv", index=False)
    (args.out / "best_params.json").write_text(json.dumps(result.best_params, indent=2))
    result.predictions.to_csv(args.out / "oof_predictions.csv", index=False)
    (args.out / "selected_features.json").write_text(json.dumps(result.features, indent=2))

    summary = scores.groupby("model")["macro_f1"].agg(["mean", "std"])

    # Two Macro F1s: pooled over every out-of-fold photo (the Step 7 report)
    # and the mean of the per-fold scores (the Step 6 selection criterion).
    label = config["splitting"]["label_column"]
    metrics = {}
    for name, preds in result.predictions.groupby("model"):
        m = compute_metrics(preds[label], preds["predicted"])
        m["pooled_macro_f1"] = m.pop("macro_f1")
        m["mean_fold_macro_f1"] = round(float(summary.loc[name, "mean"]), 4)
        metrics[name] = m
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(summary.round(3).to_string())
    if any(m in ENSEMBLE_CANDIDATES for m in args.models):
        selected = select_model(result.scores)
        (args.out / "selection.json").write_text(json.dumps(
            {"selected": selected, "mean_macro_f1": summary["mean"].round(4).to_dict()}, indent=2
        ))
        print(f"selected: {selected}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
