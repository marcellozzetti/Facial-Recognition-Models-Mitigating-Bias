"""Generate the 4-figure plot suite for every experiment in a clean run.

Walks ``<output-dir>/results.json`` and calls ``plot_experiment.py`` for
each successful experiment, writing the four PNG/PDF pairs under
``outputs/figures/<exp_label>/``.

Usage:
    python scripts/plot_all_experiments.py --runs-dir outputs/clean
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/clean"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args(argv)

    results_path = args.runs_dir / "results.json"
    if not results_path.exists():
        print(f"missing {results_path}", file=sys.stderr)
        return 1

    results = json.loads(results_path.read_text(encoding="utf-8"))
    plot_script = Path("scripts/plot_experiment.py")

    ok = 0
    for r in results:
        if r["status"] != "ok":
            continue
        label = r["label"]
        train_dir = Path(r["train_dir"])
        eval_dir = Path(r["eval_dir"])
        out_dir = args.figures_dir / label
        print(f"=== {label} ===", flush=True)
        rc = subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--train-dir",
                str(train_dir),
                "--eval-dir",
                str(eval_dir),
                "--out-dir",
                str(out_dir),
            ]
        ).returncode
        if rc != 0:
            print(f"  FAILED ({rc})", file=sys.stderr)
            continue
        ok += 1
    print(f"\nGenerated figures for {ok} experiments under {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
