"""Reanalyze HPO round1 with a Pareto-aware best-epoch criterion.

The HPO script originally picked "best by F1_macro" within each trial and
reported that single point to Optuna. With multi-objective optimisation,
that drops epochs where the IR is much lower but the F1 is fractionally
smaller — see the trials list below for concrete cases observed in round1.

This script keeps the original study.db untouched and produces a parallel
analysis from ``hpo.log``:

* Parses every ``(trial, epoch, F1, IR)`` tuple.
* For each trial, finds the **Pareto-local** epochs — i.e. the epochs not
  dominated by any other epoch *within the same trial*. When multiple
  epochs remain, picks the one with the lowest IR (favours fairness, per
  the thesis framing).
* Computes the global Pareto front using those new per-trial best points.
* Writes ``pareto_reanalyzed.json`` next to the existing artefacts.

Run after a study is complete:
    python scripts/reanalyze_hpo_round1.py \
        --log outputs/hpo/round1/hpo.log \
        --output outputs/hpo/round1/pareto_reanalyzed.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

TRIAL_RE = re.compile(
    r"trial (?P<num>\d+): hidden=\[(?P<hidden>[^\]]+)\]\s+act=(?P<act>\w+)\s+"
    r"dropout=(?P<dropout>[\d.]+)\s+norm=(?P<norm>\w+)\s+lr=(?P<lr>[\d.e+-]+)"
)
EPOCH_RE = re.compile(
    r"epoch (?P<epoch>\d+)/(?P<total>\d+)\s+val_loss=(?P<loss>[\d.]+)\s+"
    r"f1_macro=(?P<f1>[\d.]+)\s+IR=(?P<ir>[\d.]+|inf)"
)


@dataclass
class Epoch:
    epoch: int
    val_loss: float
    f1_macro: float
    inequity_rate: float  # +inf becomes 1e6 to match the script's behaviour


@dataclass
class Trial:
    number: int
    hidden_dims: list[int]
    activation: str
    dropout: float
    norm: str
    learning_rate: float
    epochs: list[Epoch]

    # populated by analysis ----
    pareto_local_epochs: list[int] = None  # type: ignore[assignment]
    best_pareto_epoch: int = -1
    best_pareto_f1: float = 0.0
    best_pareto_ir: float = float("inf")
    best_by_f1_epoch: int = -1
    best_by_f1_f1: float = 0.0
    best_by_f1_ir: float = float("inf")


def parse_log(path: Path) -> list[Trial]:
    """Parse hpo.log into a list of Trial objects in encounter order."""
    trials: list[Trial] = []
    current: Trial | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m = TRIAL_RE.search(line)
        if m:
            current = Trial(
                number=int(m["num"]),
                hidden_dims=[int(x.strip()) for x in m["hidden"].split(",")],
                activation=m["act"],
                dropout=float(m["dropout"]),
                norm=m["norm"],
                learning_rate=float(m["lr"]),
                epochs=[],
            )
            trials.append(current)
            continue

        m = EPOCH_RE.search(line)
        if m and current is not None:
            ir_raw = m["ir"]
            ir = float("inf") if ir_raw == "inf" else float(ir_raw)
            current.epochs.append(
                Epoch(
                    epoch=int(m["epoch"]),
                    val_loss=float(m["loss"]),
                    f1_macro=float(m["f1"]),
                    inequity_rate=ir if math.isfinite(ir) else 1e6,
                )
            )

    return trials


def _dominates(a: Epoch, b: Epoch) -> bool:
    """Return True iff ``a`` dominates ``b`` (a is at least as good on both
    and strictly better on one) — F1 maximised, IR minimised."""
    not_worse = a.f1_macro >= b.f1_macro and a.inequity_rate <= b.inequity_rate
    strictly_better = a.f1_macro > b.f1_macro or a.inequity_rate < b.inequity_rate
    return not_worse and strictly_better


def annotate_trial(trial: Trial) -> None:
    """Fill in best_pareto_* and best_by_f1_* on the trial."""
    if not trial.epochs:
        return

    # best by F1 (script's original criterion)
    best_f1 = max(trial.epochs, key=lambda e: e.f1_macro)
    trial.best_by_f1_epoch = best_f1.epoch
    trial.best_by_f1_f1 = best_f1.f1_macro
    trial.best_by_f1_ir = best_f1.inequity_rate

    # Pareto-local epochs within this trial
    local_pareto = [
        e for e in trial.epochs if not any(_dominates(other, e) for other in trial.epochs)
    ]
    trial.pareto_local_epochs = [e.epoch for e in local_pareto]

    # Tie-break: among Pareto-local epochs, pick the one with the lowest IR.
    # Rationale: this is a fairness-driven thesis — when several epochs
    # trade F1 for IR within the trial, we report the fairest one. This is
    # an editorial choice and is documented in the thesis methodology.
    chosen = min(local_pareto, key=lambda e: e.inequity_rate)
    trial.best_pareto_epoch = chosen.epoch
    trial.best_pareto_f1 = chosen.f1_macro
    trial.best_pareto_ir = chosen.inequity_rate


def global_pareto(values: list[tuple[float, float]]) -> list[int]:
    """Return indices of points on the global Pareto front. ``values`` is
    a list of ``(F1, IR)`` — F1 maximised, IR minimised."""
    front: list[int] = []
    for i, (f1_i, ir_i) in enumerate(values):
        dominated = False
        for j, (f1_j, ir_j) in enumerate(values):
            if i == j:
                continue
            not_worse = f1_j >= f1_i and ir_j <= ir_i
            strictly_better = f1_j > f1_i or ir_j < ir_i
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="outputs/hpo/round1/hpo.log")
    parser.add_argument("--output", default="outputs/hpo/round1/pareto_reanalyzed.json")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    # Keep only trials that have all 8 epochs (drop the reboot-orphan trial 16).
    all_trials = parse_log(log_path)
    complete = [t for t in all_trials if len(t.epochs) == 8]
    dropped = [t.number for t in all_trials if len(t.epochs) != 8]
    if dropped:
        print(f"Dropped incomplete trials (likely reboot orphans): {dropped}")

    # Deduplicate trial numbers, keeping the most recent encounter
    # (resume after reboot re-emits 'trial N: ...' on cold start).
    by_number: dict[int, Trial] = {}
    for t in complete:
        by_number[t.number] = t
    complete = sorted(by_number.values(), key=lambda t: t.number)

    for t in complete:
        annotate_trial(t)

    # Build the two Pareto fronts: by-F1 (Optuna's official) and Pareto-aware.
    by_f1_points = [(t.best_by_f1_f1, t.best_by_f1_ir) for t in complete]
    pareto_aware_points = [(t.best_pareto_f1, t.best_pareto_ir) for t in complete]

    front_by_f1 = global_pareto(by_f1_points)
    front_pareto_aware = global_pareto(pareto_aware_points)

    summary = {
        "log": str(log_path),
        "complete_trials": [t.number for t in complete],
        "dropped_trials": dropped,
        "comparison": {
            "front_by_f1": {
                "criterion": "epoch with maximum F1 within each trial (Optuna's reported value)",
                "trials": [
                    {
                        "number": complete[i].number,
                        "f1_macro": complete[i].best_by_f1_f1,
                        "inequity_rate": complete[i].best_by_f1_ir,
                        "epoch": complete[i].best_by_f1_epoch,
                    }
                    for i in front_by_f1
                ],
            },
            "front_pareto_aware": {
                "criterion": (
                    "Pareto-local non-dominated epoch within each trial, "
                    "tie-broken by lowest IR (fairness-favouring)"
                ),
                "trials": [
                    {
                        "number": complete[i].number,
                        "f1_macro": complete[i].best_pareto_f1,
                        "inequity_rate": complete[i].best_pareto_ir,
                        "epoch": complete[i].best_pareto_epoch,
                    }
                    for i in front_pareto_aware
                ],
            },
        },
        "trials": [
            {
                "number": t.number,
                "hidden_dims": t.hidden_dims,
                "activation": t.activation,
                "dropout": t.dropout,
                "norm": t.norm,
                "learning_rate": t.learning_rate,
                "best_by_f1": {
                    "epoch": t.best_by_f1_epoch,
                    "f1_macro": t.best_by_f1_f1,
                    "inequity_rate": t.best_by_f1_ir,
                },
                "best_pareto_aware": {
                    "epoch": t.best_pareto_epoch,
                    "f1_macro": t.best_pareto_f1,
                    "inequity_rate": t.best_pareto_ir,
                    "pareto_local_epochs": t.pareto_local_epochs,
                },
                "epoch_history": [asdict(e) for e in t.epochs],
            }
            for t in complete
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # Pretty-print the comparison
    print(f"\n=== Pareto front (best-by-F1 criterion — Optuna official): "
          f"{len(front_by_f1)} trials ===")
    for i in front_by_f1:
        t = complete[i]
        print(
            f"  trial #{t.number:2d}  "
            f"F1={t.best_by_f1_f1:.4f}  "
            f"IR={t.best_by_f1_ir:.4f}  "
            f"epoch={t.best_by_f1_epoch}  "
            f"({t.hidden_dims} {t.activation} drop={t.dropout:.2f} norm={t.norm} lr={t.learning_rate:.2e})"
        )

    print(f"\n=== Pareto front (Pareto-aware best-epoch criterion): "
          f"{len(front_pareto_aware)} trials ===")
    for i in front_pareto_aware:
        t = complete[i]
        print(
            f"  trial #{t.number:2d}  "
            f"F1={t.best_pareto_f1:.4f}  "
            f"IR={t.best_pareto_ir:.4f}  "
            f"epoch={t.best_pareto_epoch}  "
            f"({t.hidden_dims} {t.activation} drop={t.dropout:.2f} norm={t.norm} lr={t.learning_rate:.2e})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
