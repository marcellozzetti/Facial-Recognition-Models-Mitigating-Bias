"""Optuna multi-objective HPO over the MLP classification head (diretriz nº 3).

The orientador (kickoff 2026-05-11) asked us to fix the backbone
(ResNet50) and let Optuna search the topology of the dense head plus a
small set of training knobs (learning rate, weight decay). The trial
objective is bi-criterion:

* maximise ``val_f1_macro`` (utility);
* minimise ``val_inequity_rate`` (fairness — max F1 / min F1 across the
  7 FairFace races; 1.0 means perfect parity).

Implementation notes:

- We start from a *baseline experiment* config (default: Exp 5 — CE +
  AdamW + Cosine + dropout=0.2, our cleanest run from 2026-05-09). Only
  the model-head fields and ``training.learning_rate`` are overridden by
  each trial; everything else (dataset, split, batch size, scheduler) is
  kept identical so trials are comparable.
- Trials run for ``--hpo-epochs`` (default 8). Full 25-epoch confirms run
  only for the Pareto-front winners after the study completes.
- Optuna persistence is a local SQLite file (``study.db`` in the output
  dir) so the study is resumable. Pass ``--n-trials 0`` to just
  re-summarise an existing study without running new trials.
- The search runs in-process: we re-use ``setup_dataset`` once and pass
  the same loaders to every trial to avoid the 5-min preprocessing cost
  per trial.

Usage:
    # First-round HPO (20 trials, 8 epochs each)
    python scripts/hpo_head.py --n-trials 20 --hpo-epochs 8 \
        --base-config configs/experiments/exp05_ce_adamw_cosine.yaml \
        --output-dir outputs/hpo/round1

    # Sanity check (2 trials, 2 epochs) — use before committing to a long run
    python scripts/hpo_head.py --n-trials 2 --hpo-epochs 2 \
        --output-dir outputs/hpo/sanity

    # Re-summarise an existing study (no new trials)
    python scripts/hpo_head.py --n-trials 0 --output-dir outputs/hpo/round1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.evaluation.metrics import classification_metrics, fairness_audit
from face_bias.models.resnet import LResNet50E_IR
from face_bias.training.optimizers import build_optimizer
from face_bias.training.schedulers import build_scheduler, is_step_per_batch
from face_bias.training.trainer import build_loss
from face_bias.utils.logging import make_run_id, setup_logging
from face_bias.utils.reproducibility import seed_from_config

# Width buckets we let Optuna pick from per hidden layer. Restricting to a
# discrete set keeps the search space small and the resulting topologies
# readable (e.g., "1024->256" instead of "987->213").
HIDDEN_WIDTH_CHOICES = [128, 256, 512, 1024, 2048]
ACTIVATION_CHOICES = ["relu", "gelu", "silu"]
NORM_CHOICES = ["none", "batchnorm", "layernorm"]


# ----------------------------------------------------------- helpers


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _suggest_head_params(trial) -> dict[str, Any]:
    """Sample one MLP topology + LR. All other knobs come from the base config."""
    depth = trial.suggest_int("mlp_depth", 1, 3)
    hidden_dims: list[int] = []
    for i in range(depth):
        # categorical over a fixed grid; Optuna lets each layer pick
        # independently so funnel/inverted-funnel/uniform shapes are reachable.
        width = trial.suggest_categorical(f"mlp_hidden_{i}", HIDDEN_WIDTH_CHOICES)
        hidden_dims.append(int(width))
    return {
        "mlp_hidden_dims": hidden_dims,
        "mlp_activation": trial.suggest_categorical("mlp_activation", ACTIVATION_CHOICES),
        "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.6),
        "mlp_norm": trial.suggest_categorical("mlp_norm", NORM_CHOICES),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def _apply_trial_to_config(base: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with the trial's head params applied."""
    import copy

    cfg = copy.deepcopy(base)
    cfg["model"]["head"] = "mlp"
    cfg["model"]["mlp_hidden_dims"] = params["mlp_hidden_dims"]
    cfg["model"]["mlp_activation"] = params["mlp_activation"]
    cfg["model"]["mlp_dropout"] = params["mlp_dropout"]
    cfg["model"]["mlp_norm"] = params["mlp_norm"]
    cfg["training"]["learning_rate"] = params["learning_rate"]
    # CE — the head is a plain classifier, so ArcFace loss is meaningless here.
    cfg["training"]["loss_function"] = "cross_entropy"
    return cfg


# ----------------------------------------------------------- training core


def _build_model(cfg: dict[str, Any], num_classes: int, device: torch.device) -> torch.nn.Module:
    mcfg = cfg["model"]
    model = LResNet50E_IR(
        num_classes=num_classes,
        dropout=mcfg["dropout"],
        head="mlp",
        pretrained=mcfg.get("pretrained", True),
        mlp_hidden_dims=mcfg["mlp_hidden_dims"],
        mlp_activation=mcfg["mlp_activation"],
        mlp_dropout=mcfg["mlp_dropout"],
        mlp_norm=mcfg["mlp_norm"],
    )
    return model.to(device)


@torch.no_grad()
def _evaluate(model, loader, loss_fn, device, class_names: list[str]) -> dict[str, Any]:
    import numpy as np

    model.eval()
    preds, targets, probas = [], [], []
    running, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        running += loss.item() * images.size(0)
        seen += images.size(0)
        proba = torch.softmax(logits, dim=1)
        probas.append(proba.cpu().numpy())
        preds.append(proba.argmax(dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    y_proba = np.concatenate(probas)

    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)
    metrics["loss"] = running / max(seen, 1)
    metrics["fairness"] = fairness_audit(y_true, y_pred, class_names)
    return metrics


def _pareto_local_best(history: list[dict]) -> dict:
    """Pick the epoch this trial should report to Optuna.

    Returns the epoch on the trial's *local* Pareto front (F1 maximised,
    IR minimised) that has the **lowest IR**, breaking ties in favour of
    fairness — the thesis is fairness-driven, so when several epochs are
    non-dominated within the trial we surface the fairest one.

    The naive "epoch with max F1" criterion silently discards epochs
    where IR is much lower but F1 is fractionally smaller. Round1 of the
    HPO study showed this regularly — trials 8, 12 and 13 only entered
    the global Pareto front after re-analysing with this criterion.
    """
    if not history:
        raise ValueError("history is empty")

    def dominates(a: dict, b: dict) -> bool:
        not_worse = a["val_f1_macro"] >= b["val_f1_macro"] and a["val_inequity_rate"] <= b["val_inequity_rate"]
        strictly_better = a["val_f1_macro"] > b["val_f1_macro"] or a["val_inequity_rate"] < b["val_inequity_rate"]
        return not_worse and strictly_better

    local_pareto = [h for h in history if not any(dominates(other, h) for other in history)]
    return min(local_pareto, key=lambda h: h["val_inequity_rate"])


def _train_trial(
    cfg: dict[str, Any],
    dataloaders: dict,
    class_names: list[str],
    epochs: int,
    device: torch.device,
    trial=None,
) -> dict[str, Any]:
    """Run a single Optuna trial end-to-end and return its Pareto-aware best."""
    num_classes = len(class_names)
    model = _build_model(cfg, num_classes, device)
    loss_fn = build_loss(cfg).to(device)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(dataloaders["train"]))

    grad_clip = cfg["training"].get("grad_clip_norm")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in dataloaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if is_step_per_batch(scheduler):
                scheduler.step()
        if not is_step_per_batch(scheduler):
            scheduler.step()

        val_metrics = _evaluate(model, dataloaders["val"], loss_fn, device, class_names)
        f1m = val_metrics["f1_macro"]
        ir = val_metrics["fairness"]["f1"]["inequity_rate"]
        # Optuna can't deal with +inf; substitute a large finite penalty so
        # collapsing trials are still comparable but dominated.
        ir_finite = ir if math.isfinite(ir) else 1e6
        logging.info(
            f"  epoch {epoch}/{epochs}  val_loss={val_metrics['loss']:.4f}  "
            f"f1_macro={f1m:.4f}  IR={ir:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_f1_macro": f1m,
                "val_inequity_rate": ir_finite,
            }
        )
        # Optuna's Trial.report / should_prune is single-objective only;
        # multi-objective studies have no built-in pruning, so we just run
        # the trial to completion (kept short via --hpo-epochs).

    best = _pareto_local_best(history)
    # Stash the full epoch history on the Optuna trial so post-hoc
    # analysis (and the next round's reanalyzer) can recompute fronts
    # without re-parsing logs.
    if trial is not None:
        trial.set_user_attr("epoch_history", history)
        trial.set_user_attr("best_epoch", best["epoch"])

    return {
        "val_f1_macro": best["val_f1_macro"],
        "val_inequity_rate": best["val_inequity_rate"],
        "val_loss": best["val_loss"],
        "epoch_of_best": best["epoch"],
    }


# ----------------------------------------------------------- CLI / study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna HPO over the MLP classification head.")
    parser.add_argument(
        "--base-config",
        default="configs/experiments/exp05_ce_adamw_cosine.yaml",
        help="Baseline config to inherit dataset/split/optimizer/scheduler from.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hpo/run",
        help="Where to write study.db, best_params.json, and trials.csv.",
    )
    parser.add_argument(
        "--study-name",
        default="mlp_head_hpo",
        help="Optuna study name (also used as the SQLite key — keep stable across resumes).",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--hpo-epochs",
        type=int,
        default=8,
        help="Epochs per trial (kept short; refit best trial separately at 25 epochs).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import optuna
    except ImportError:
        print(
            "optuna is not installed. Install with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install 'face-bias[hpo]'",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base_config)
    # Redirect logs to the HPO output dir so a study has its own log file
    # next to its study.db / trials.csv.
    base_cfg["logging"]["log_dir"] = str(output_dir)
    base_cfg["logging"]["log_training_file"] = "hpo.log"
    run_id = make_run_id()
    setup_logging(base_cfg, "log_training_file", run_id=run_id)

    # Lock seed/epochs from the CLI so the study is reproducible.
    base_cfg["training"]["random_state"] = args.seed
    base_cfg["training"]["num_epochs"] = args.hpo_epochs
    seed_from_config({"training": base_cfg["training"]})

    device = _resolve_device(args.device)
    logging.info(f"HPO base config: {args.base_config}")
    logging.info(f"device={device}  n_trials={args.n_trials}  hpo_epochs={args.hpo_epochs}")

    # Build the dataloaders once; every trial reuses them.
    dataloaders, label_encoder, num_classes = setup_dataset(base_cfg)
    class_names = list(label_encoder.classes_)
    logging.info(f"{num_classes} classes: {class_names}")
    logging.info(
        f"train={len(dataloaders['train'].dataset)}  "
        f"val={len(dataloaders['val'].dataset)}  "
        f"test={len(dataloaders['test'].dataset)}"
    )

    storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "minimize"],  # F1_macro up, IR down
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial):
        params = _suggest_head_params(trial)
        cfg = _apply_trial_to_config(base_cfg, params)
        logging.info(
            f"trial {trial.number}: hidden={params['mlp_hidden_dims']}  "
            f"act={params['mlp_activation']}  dropout={params['mlp_dropout']:.2f}  "
            f"norm={params['mlp_norm']}  lr={params['learning_rate']:.2e}"
        )
        result = _train_trial(
            cfg, dataloaders, class_names, args.hpo_epochs, device, trial=trial
        )
        return result["val_f1_macro"], result["val_inequity_rate"]

    if args.n_trials > 0:
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # ---- summarise ----
    trials_df = study.trials_dataframe(attrs=("number", "values", "params", "state"))
    trials_csv = output_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    logging.info(f"wrote {trials_csv}")

    pareto = study.best_trials
    pareto_summary = [
        {
            "number": t.number,
            "f1_macro": t.values[0],
            "inequity_rate": t.values[1],
            "params": t.params,
        }
        for t in pareto
    ]
    best_path = output_dir / "best_params.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": args.study_name,
                "base_config": args.base_config,
                "hpo_epochs": args.hpo_epochs,
                "pareto_front": pareto_summary,
            },
            f,
            indent=2,
        )
    logging.info(f"wrote {best_path}")
    logging.info(f"Pareto front size: {len(pareto)}")
    for t in pareto:
        logging.info(
            f"  trial #{t.number}  f1_macro={t.values[0]:.4f}  IR={t.values[1]:.3f}  {t.params}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
