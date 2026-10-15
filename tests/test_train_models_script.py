import importlib.util
import json
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_models.py"


def _run(tmp_path, feature_table, config):
    csv = tmp_path / "features_clean.csv"
    feature_table.to_csv(csv, index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    out = tmp_path / "results"
    spec = importlib.util.spec_from_file_location("train_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv
    sys.argv = ["train_models.py", "--config", str(cfg), "--features", str(csv),
                "--out", str(out), "--models", "random_forest", "--n-trials", "1"]
    try:
        module.main()
    finally:
        sys.argv = argv
    return out


def test_metrics_name_pooled_and_mean_fold_macro_f1_apart(tmp_path, feature_table, config):
    out = _run(tmp_path, feature_table, config)
    m = json.loads((out / "metrics.json").read_text())["random_forest"]
    selection = json.loads((out / "selection.json").read_text())
    assert "macro_f1" not in m
    assert 0.0 <= m["pooled_macro_f1"] <= 1.0
    # the per-fold mean is the selection criterion and must match selection.json
    assert m["mean_fold_macro_f1"] == selection["mean_macro_f1"]["random_forest"]


def test_selected_features_written_per_fold(tmp_path, feature_table, config):
    out = _run(tmp_path, feature_table, config)
    features = json.loads((out / "selected_features.json").read_text())
    assert features == [["feat_1", "feat_2"]] * 5
