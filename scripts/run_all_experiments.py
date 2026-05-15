"""Run all 11 MBA-replication experiments and collect their metrics.

Usage:
    # Smoke run — 5 epochs each, ~4h on RTX 4070 SUPER
    python scripts/run_all_experiments.py --epochs 5 --output-dir outputs/smoke

    # Full clean run — keep config-declared num_epochs (25/40)
    python scripts/run_all_experiments.py --output-dir outputs/clean

    # Subset
    python scripts/run_all_experiments.py --only exp01_ce_sgd_onecycle exp03_ce_adamw_onecycle
    python scripts/run_all_experiments.py --skip exp06_arcface_adamw_cosine

The script:
1. Iterates configs/experiments/exp*.yaml in sorted order.
2. For each, calls face-bias-train + face-bias-evaluate as subprocesses
   so a crash in one experiment doesn't poison the rest.
3. Persists partial results to ``<output-dir>/results.json`` after each
   experiment (resume-friendly).
4. Renders a comparison table at ``<output-dir>/comparison_table.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CONFIGS_DIR = Path("configs/experiments")


def find_exp_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("exp*.yaml"))


def _latest_subdir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    children = [p for p in parent.iterdir() if p.is_dir()]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def train_and_evaluate(
    config: Path,
    output_root: Path,
    epochs: int | None,
    device: str,
) -> dict[str, Any]:
    label = config.stem
    train_root = output_root / label / "train"
    eval_root = output_root / label / "evaluate"
    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {label} — training ===", flush=True)
    train_cmd = [
        "face-bias-train",
        "--config",
        str(config),
        "--output-dir",
        str(train_root),
        "--device",
        device,
    ]
    if epochs is not None:
        train_cmd.extend(["--epochs", str(epochs)])

    train_start = time.time()
    train_rc = subprocess.run(train_cmd).returncode
    train_seconds = time.time() - train_start
    if train_rc != 0:
        return {
            "label": label,
            "status": "failed",
            "stage": "train",
            "returncode": train_rc,
            "train_seconds": train_seconds,
        }

    train_dir = _latest_subdir(train_root)
    if train_dir is None:
        return {"label": label, "status": "failed", "stage": "train", "error": "no run dir"}
    checkpoint = train_dir / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        return {"label": label, "status": "failed", "stage": "train", "error": "no best.pt"}

    print(f"=== {label} — evaluating ===", flush=True)
    eval_cmd = [
        "face-bias-evaluate",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(eval_root),
        "--device",
        device,
        "--split",
        "test",
    ]
    eval_start = time.time()
    eval_rc = subprocess.run(eval_cmd).returncode
    eval_seconds = time.time() - eval_start
    if eval_rc != 0:
        return {
            "label": label,
            "status": "failed",
            "stage": "evaluate",
            "returncode": eval_rc,
            "train_dir": str(train_dir),
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        }

    eval_dir = _latest_subdir(eval_root)
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    fairness = json.loads((eval_dir / "fairness_audit.json").read_text(encoding="utf-8"))

    history_path = train_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    last_epoch = history[-1] if history else {}

    return {
        "label": label,
        "status": "ok",
        "config": str(config),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "fairness": fairness,
        "last_epoch": last_epoch,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }


def write_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Render a Markdown comparison table covering all the runs."""
    lines = [
        "# MBA Experiment Replication — Comparison Table",
        "",
        "Aggregate metrics on the test split (best.pt = lowest val_loss).",
        "",
        "| # | Experiment | Status | Acc | F1 | Precision | IR (F1) | Gap (F1) | Train (min) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results, start=1):
        label = r["label"]
        status = r["status"]
        if status != "ok":
            stage = r.get("stage", "?")
            lines.append(f"| {i} | `{label}` | **FAILED ({stage})** | | | | | | |")
            continue
        m = r["metrics"]
        f = r["fairness"]["f1"]
        per_class = r["fairness"]["f1"]
        train_min = r["train_seconds"] / 60
        precision_macro = (
            r["fairness"]["precision"]["mean"]
            if "precision" in r["fairness"]
            else m.get("f1_macro", 0)
        )
        lines.append(
            f"| {i} | `{label}` | ok | "
            f"{m['accuracy']:.3f} | {m['f1_macro']:.3f} | "
            f"{precision_macro:.3f} | "
            f"{f['inequity_rate']:.3f} | {per_class['max_min_disparity']:.3f} | "
            f"{train_min:.1f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config num_epochs (smoke runs)"
    )
    parser.add_argument("--device", default="cuda", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--output-dir", default="outputs/runs", type=Path)
    parser.add_argument(
        "--configs-dir",
        default=CONFIGS_DIR,
        type=Path,
        help="Directory holding exp*.yaml configs to run (default: configs/experiments)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Only run these label stems")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip these label stems")
    args = parser.parse_args(argv)

    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    results_path = output_root / "results.json"
    if results_path.exists():
        results: list[dict[str, Any]] = json.loads(results_path.read_text(encoding="utf-8"))
        completed_labels = {r["label"] for r in results}
        print(f"Resuming with {len(completed_labels)} completed experiment(s).")
    else:
        results = []
        completed_labels = set()

    configs = sorted(args.configs_dir.glob("exp*.yaml"))
    if not configs:
        print(f"No configs under {args.configs_dir} matching exp*.yaml", file=sys.stderr)
        return 1

    for config in configs:
        label = config.stem
        if label in completed_labels:
            print(f"Skipping {label} — already in results.json")
            continue
        if label in args.skip:
            print(f"Skipping {label} (--skip)")
            continue
        if args.only is not None and label not in args.only:
            continue

        result = train_and_evaluate(config, output_root, args.epochs, args.device)
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"=== {label}: {result['status']} "
            f"(train={result.get('train_seconds', 0):.0f}s, "
            f"eval={result.get('eval_seconds', 0):.0f}s) ==="
        )

    report_path = output_root / "comparison_table.md"
    write_report(results, report_path)
    print(f"\n=== Done. Report: {report_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
