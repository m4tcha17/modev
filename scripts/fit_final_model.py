"""Run the post-selection steps (process.md Steps 8-9, REQ-4.2-7):
re-tune the selected ensemble on the full cleaned dataset (grouped 5-fold),
train it on every photo plus augmented copies, save the deployable artifact,
and write the TreeSHAP feature-family attribution for that model.

Needs train_models.py's selection.json. The deployed model's quality is the
nested-CV estimate from train_models.py; nothing here scores it again.

Usage:
  uv run python scripts/fit_final_model.py
  uv run python scripts/fit_final_model.py --n-trials 3      # smoke run
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from copra_grading.artifact import save_artifact
from copra_grading.explainability import compute_aggregate_shap_by_family, family_shares, feature_family_map
from copra_grading.features.table import ID_COLUMNS, feature_columns, originals
from copra_grading.selection import fit_final_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_clean.csv"))
    parser.add_argument("--results", type=Path, default=Path("data/processed/results"))
    parser.add_argument("--out", type=Path, default=None, help="artifact directory (default: config artifact.path)")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    n_trials = args.n_trials or config["tuning"]["n_trials"]
    out = args.out or Path(config["artifact"]["path"])
    selection = json.loads((args.results / "selection.json").read_text())
    selected = selection["selected"]
    table = pd.read_csv(args.features, dtype={c: str for c in ID_COLUMNS})
    features = feature_columns(table)

    print(f"re-tuning {selected} on {table['id'].nunique()} photos, {n_trials} trials", flush=True)
    model, params = fit_final_model(table, config, selected, n_trials)

    extra = {
        "fitted": date.today().isoformat(),
        "training_photos": int(table["id"].nunique()),
        "nested_cv_mean_macro_f1": selection["mean_macro_f1"],
    }
    save_artifact(model, features, config, out, params=params, extra=extra)
    print(f"artifact -> {out}")

    X = originals(table)[features]
    shap_raw = compute_aggregate_shap_by_family(model, X, feature_family_map(features))
    report = {"model": selected, "rows": len(X), "mean_abs_shap": shap_raw, "share": family_shares(shap_raw)}
    (args.results / "shap_by_family.json").write_text(json.dumps(report, indent=2))
    for family, share in sorted(report["share"].items(), key=lambda kv: -kv[1]):
        print(f"  {family}: {share:.1%} of mean |SHAP|")
    print(f"shap -> {args.results / 'shap_by_family.json'}")


if __name__ == "__main__":
    main()
