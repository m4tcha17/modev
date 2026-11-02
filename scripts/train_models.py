"""Run Steps 5-7 on the cleaned feature table from prepare_folds.py: tune all
four models with Optuna on Macro F1 (5), compare them across the 5 folds and
pick the best ensemble (6), report held-out metrics per photo (7).

Folds are rebuilt with folds.make_folds - same split, same augmented copies
as prepare_folds.py. Each model is also scored once with library-default
params as a sanity reference for how much tuning bought.

Note: tuning and the reported scores use the same 5 folds (not nested CV),
so tuned scores are optimistic by however much the search fit the folds.
Compare against the default-param scores.

Usage:
  uv run python scripts/train_models.py                       # config n_trials, 4 threads
  uv run python scripts/train_models.py --trials 5 --n-jobs 2 # quick smoke run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from copra_grading.evaluation import compute_metrics, fold_scores
from copra_grading.features.table import ID_COLUMNS
from copra_grading.folds import make_folds
from copra_grading.models import MODEL_NAMES
from copra_grading.models.tuning import tune_model
from copra_grading.selection import ENSEMBLE_CANDIDATES, cross_validate_models, select_model


def _md(df: pd.DataFrame) -> str:
    """DataFrame as a markdown table (no tabulate dependency)."""
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in r] for i, r in zip(df.index, df.to_numpy())]
    return "\n".join("| " + " | ".join(r) + " |" for r in [head, ["---"] * len(head), *rows])


def _fmt(scores: list[float]) -> str:
    return f"{np.mean(scores):.3f} ± {np.std(scores):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--trials", type=int, default=None, help="override tuning.n_trials")
    parser.add_argument("--n-jobs", type=int, default=4, help="threads per model fit")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    n_trials = args.trials if args.trials is not None else config["tuning"]["n_trials"]
    seed = config["splitting"]["random_state"]
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    folds = make_folds(table, config)
    args.out.mkdir(parents=True, exist_ok=True)
    classes = sorted(table["copra_class"].unique())
    print(f"{table['id'].nunique()} photos, classes {classes}, {len(folds)} folds, "
          f"{n_trials} trials/model, n_jobs={args.n_jobs}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tuned, default_oof = {}, {}
    for name in args.models:
        t0 = time.time()
        default_oof[name] = cross_validate_models(folds, {name: {}}, args.n_jobs)[name]
        study = tune_model(name, folds, n_trials, n_jobs=args.n_jobs, seed=seed)
        tuned[name] = study.best_params
        print(f"[{name}] default {_fmt(fold_scores(default_oof[name]))}  "
              f"tuned best {study.best_value:.3f}  ({time.time() - t0:.0f}s)")
        study.trials_dataframe().to_csv(args.out / f"trials_{name}.csv", index=False)

    oof = cross_validate_models(folds, tuned, args.n_jobs)
    scores = {name: fold_scores(df) for name, df in oof.items()}
    selected = select_model(scores) if set(scores) & set(ENSEMBLE_CANDIDATES) else None

    summary = {"n_photos": int(table["id"].nunique()), "classes": classes,
               "n_trials": n_trials, "selected": selected, "models": {}}
    lines = ["# Model results", "",
             f"{summary['n_photos']} photos, classes {', '.join(classes)}, 5 folds "
             f"(StratifiedGroupKFold by batch_id), {n_trials} Optuna trials/model.", "",
             "| Model | Macro F1 tuned (mean ± sd) | Macro F1 default | Accuracy (pooled) | Per-fold Macro F1 |",
             "|---|---|---|---|---|"]
    for name in oof:
        m = compute_metrics(oof[name]["y_true"], oof[name]["y_pred"])
        d = fold_scores(default_oof[name])
        oof[name].to_csv(args.out / f"oof_{name}.csv", index=False)
        summary["models"][name] = {
            "params": tuned[name], "fold_macro_f1": scores[name],
            "mean_macro_f1": float(np.mean(scores[name])),
            "default_fold_macro_f1": d, "pooled_accuracy": m["accuracy"],
            "pooled_macro_f1": m["macro_f1"],
            "per_class": m["per_class"].reset_index().to_dict(orient="records"),
            "confusion_matrix": m["confusion_matrix"].to_dict(orient="split"),
        }
        tag = " **selected**" if name == selected else (" (baseline)" if name == "logistic_regression" else "")
        lines.append(f"| {name}{tag} | {_fmt(scores[name])} | {_fmt(d)} | {m['accuracy']:.3f} | "
                     + ", ".join(f"{s:.3f}" for s in scores[name]) + " |")
    for name in oof:
        m = compute_metrics(oof[name]["y_true"], oof[name]["y_pred"])
        lines += ["", f"## {name}", "", "Per class (pooled over held-out folds):", "",
                  _md(m["per_class"].round(3)), "",
                  "Confusion matrix (rows true, cols predicted, moisture order):", "",
                  _md(m["confusion_matrix"]), "", f"Params: `{tuned[name]}`"]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + len(oof)]))
    print(f"\nselected: {selected}\nfull report -> {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
