"""One launcher for every pipeline process.

Usage:
  .\\run                      menu: pick a process by number
  .\\run train                start one process directly
  .\\run train --n-trials 3   extra arguments go to that process's script
  .\\run all                  features -> folds -> train, stops at the first failure

Without run.bat:  uv run python run.py [process] [args...]
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# name -> (script, description). Menu order = pipeline order.
PROCESSES = {
    "mask": ("mask_photos.py", "Background masking: masked photos + red-outline overlays for checking by eye"),
    "features": ("extract_features.py", "Feature extraction (Steps 1-3) -> data/processed/features.csv"),
    "folds": ("prepare_folds.py", "Outlier removal + folds (Step 4) -> features_clean.csv, folds.csv"),
    "train": ("train_models.py", "Model training, selection, metrics (Steps 5-7), ~80 min -> data/processed/results/"),
    "quick-train": ("train_models.py", "Smoke test of training: 3 trials per model, a few minutes"),
}
PIPELINE = ("features", "folds", "train")
QUICK_TRAIN_ARGS = ["--n-trials", "3"]


def run_process(name: str, extra: list[str]) -> int:
    """Run one process's script with live output. Returns its exit code."""
    script, description = PROCESSES[name]
    args = (QUICK_TRAIN_ARGS if name == "quick-train" else []) + extra
    print(f"\n=== {name}: {description}", flush=True)
    start = time.time()
    code = subprocess.call([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT)
    minutes, seconds = divmod(int(time.time() - start), 60)
    status = "done" if code == 0 else f"FAILED (exit code {code})"
    print(f"=== {name} {status} in {minutes}m {seconds}s", flush=True)
    return code


def run_all(extra: list[str]) -> int:
    for name in PIPELINE:
        code = run_process(name, extra if name == "train" else [])
        if code != 0:
            print(f"stopped: {name} failed, later steps skipped")
            return code
    return 0


def menu() -> str | None:
    """Ask for a process by number. None = quit."""
    choices = [*PROCESSES, "all"]
    print("Copra pipeline - choose a process:\n")
    for i, name in enumerate(choices, 1):
        desc = PROCESSES[name][1] if name in PROCESSES else "Run features -> folds -> train in order"
        print(f"  {i}. {name:<12} {desc}")
    print("  q. quit\n")
    while True:
        try:
            answer = input("number: ").strip().lstrip("﻿").lower()
        except EOFError:  # no terminal input left (e.g. piped)
            return None
        if answer in ("q", ""):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        print(f"enter 1-{len(choices)} or q")


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv:
        name, extra = argv[0], argv[1:]
    else:
        name, extra = menu(), []
        if name is None:
            return 0
    if name == "all":
        return run_all(extra)
    if name not in PROCESSES:
        print(f"unknown process '{name}'. Choose from: {', '.join([*PROCESSES, 'all'])}")
        return 2
    return run_process(name, extra)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
