import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_launcher", ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_every_process_points_at_an_existing_script():
    for script, _ in run.PROCESSES.values():
        assert (ROOT / "scripts" / script).is_file()


def test_pipeline_runs_in_order_and_stops_at_first_failure(monkeypatch):
    calls = []

    def fake(name, extra):
        calls.append((name, extra))
        return 1 if name == "folds" else 0

    monkeypatch.setattr(run, "run_process", fake)
    assert run.run_all(["--n-trials", "3"]) == 1
    assert calls == [("features", []), ("folds", [])]


def test_extra_args_go_to_train_only(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "run_process", lambda name, extra: calls.append((name, extra)) or 0)
    assert run.run_all(["--n-trials", "3"]) == 0
    assert calls == [("features", []), ("folds", []), ("train", ["--n-trials", "3"])]


def test_unknown_process_is_rejected(capsys):
    assert run.main(["nonsense"]) == 2
    assert "unknown process" in capsys.readouterr().out


def test_menu_accepts_number_and_name(monkeypatch):
    answers = iter(["9", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert run.menu() == "train"
    monkeypatch.setattr("builtins.input", lambda _: "folds")
    assert run.menu() == "folds"
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert run.menu() is None
