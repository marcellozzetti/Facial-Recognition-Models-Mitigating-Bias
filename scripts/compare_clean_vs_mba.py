"""Print a side-by-side table comparing the clean-run results to the MBA report."""

from __future__ import annotations

import json
from pathlib import Path

MBA_REPORTED_ACC = {
    "exp01_ce_sgd_onecycle": 0.68,
    "exp02_arcface_sgd_onecycle": 0.61,
    "exp03_ce_adamw_onecycle": 0.67,
    "exp04_arcface_adamw_onecycle": 0.58,
    "exp05_ce_adamw_cosine": 0.62,
    "exp06_arcface_adamw_cosine": None,  # MBA reported zeros
    "exp07_ce_adamw_onecycle_blackwhite": 0.95,
    "exp08_arcface_adamw_onecycle_blackwhite": 0.94,
    "exp09_ce_adamw_onecycle_dropout05": 0.67,
    "exp10_arcface_adamw_onecycle_dropout05": 0.59,
    "exp11_ce_adamw_onecycle_40ep": 0.67,
}


def main() -> None:
    results_path = Path("outputs/clean/results.json")
    if not results_path.exists():
        raise SystemExit(f"missing {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    header = (
        f"{'#':<3} {'Experiment':<46s} "
        f"{'MBA':>6s}  {'Clean':>6s}  {'Delta':>7s}  "
        f"{'IR':>6s}  {'Gap':>5s}  {'Train(min)':>10s}"
    )
    print(header)
    print("-" * len(header))

    total_train = 0.0
    for r in results:
        if r["status"] != "ok":
            print(f"{r['label']:<46s}  FAILED ({r.get('stage','?')})")
            continue
        label = r["label"]
        n = label.split("_")[0].replace("exp", "")
        acc = r["metrics"]["accuracy"]
        ir = r["fairness"]["f1"]["inequity_rate"]
        gap = r["fairness"]["f1"]["max_min_disparity"]
        train_min = r["train_seconds"] / 60
        total_train += train_min
        mba = MBA_REPORTED_ACC.get(label)

        if mba is None:
            mba_str = "(0)"
            delta_str = "(new)"
        else:
            mba_str = f"{mba:.2f}"
            delta_str = f"{acc - mba:+.3f}"

        ir_str = "inf" if ir == float("inf") else f"{ir:.2f}"
        print(
            f"{n:<3} {label:<46s} "
            f"{mba_str:>6s}  {acc:>6.3f}  {delta_str:>7s}  "
            f"{ir_str:>6s}  {gap:>5.2f}  {train_min:>10.1f}"
        )

    print()
    print(f"Total wall-clock train time: {total_train:.1f} min ({total_train/60:.2f} h)")


if __name__ == "__main__":
    main()
