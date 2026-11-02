"""Cataloga experimentos em outputs/experiment_log.json.

Uso standalone:
    python -I pipelines/log_experiment.py \\
        --exp-id "A_s42_v1" \\
        --pipeline "05_train_ablation" \\
        --config-id A --seed 42 --max-epochs 15 --patience 5 \\
        --best-epoch 6 --f1 0.7106 --dr 0.6042 --worst-f1 0.5330 \\
        --checkpoint outputs/etapa2/config_A/A_seed42.pt \\
        --notes "Baseline protocolo original"

Ou importado por outros pipelines após o treino.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "outputs" / "experiment_log.json"


def load_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "experiments": []}


def save_log(log: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def add_entry(
    exp_id: str,
    pipeline: str,
    config: dict,
    results: dict,
    outputs: dict,
    notes: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Adiciona entrada ao log. Retorna a entrada criada."""
    log = load_log()

    # Verifica duplicata
    existing_ids = {e["id"] for e in log["experiments"]}
    if exp_id in existing_ids:
        print(f"[log_experiment] AVISO: exp_id '{exp_id}' já existe — sobrescrevendo.")
        log["experiments"] = [e for e in log["experiments"] if e["id"] != exp_id]

    entry = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "config": config,
        "results": results,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "notes": notes,
        "tags": tags or [],
    }

    log["experiments"].append(entry)
    save_log(log)
    print(f"[log_experiment] Catalogado: {exp_id} -> {LOG_PATH}")
    return entry


def list_experiments(filter_pipeline: str | None = None) -> list[dict]:
    log = load_log()
    exps = log["experiments"]
    if filter_pipeline:
        exps = [e for e in exps if e["pipeline"] == filter_pipeline]
    return exps


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cataloga um experimento no log JSON.")
    p.add_argument("--exp-id", required=True, help="ID único, ex: A_s42_v1")
    p.add_argument("--pipeline", required=True, help="Nome do pipeline, ex: 05_train_ablation")
    # Config fields
    p.add_argument("--config-id", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--extra-config", default="{}", help="JSON adicional para config")
    # Results
    p.add_argument("--best-epoch", type=int, default=None)
    p.add_argument("--f1", type=float, default=None, help="F1-macro best checkpoint")
    p.add_argument("--dr", type=float, default=None, help="Disparity Ratio best checkpoint")
    p.add_argument("--worst-f1", type=float, default=None, help="Worst-class F1 best checkpoint")
    p.add_argument("--extra-results", default="{}", help="JSON adicional para results")
    # Outputs
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--history", default=None)
    p.add_argument("--plots-dir", default=None)
    # Meta
    p.add_argument("--notes", default="")
    p.add_argument("--tags", nargs="*", default=[])

    args = p.parse_args(argv)

    config = {k: v for k, v in {
        "config_id":  args.config_id,
        "seed":       args.seed,
        "max_epochs": args.max_epochs,
        "patience":   args.patience,
        "batch_size": args.batch_size,
        "dataset":    args.dataset,
    }.items() if v is not None}
    config.update(json.loads(args.extra_config))

    results = {k: v for k, v in {
        "best_epoch":     args.best_epoch,
        "f1_macro":       args.f1,
        "disparity_ratio": args.dr,
        "worst_class_f1": args.worst_f1,
    }.items() if v is not None}
    results.update(json.loads(args.extra_results))

    outputs = {k: v for k, v in {
        "checkpoint": args.checkpoint,
        "history":    args.history,
        "plots_dir":  args.plots_dir,
    }.items() if v is not None}

    add_entry(
        exp_id=args.exp_id,
        pipeline=args.pipeline,
        config=config,
        results=results,
        outputs=outputs,
        notes=args.notes,
        tags=args.tags,
    )


if __name__ == "__main__":
    main()
