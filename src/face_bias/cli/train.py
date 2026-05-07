"""CLI: train a face-recognition model on the configured dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.models.resnet import LResNet50E_IR
from face_bias.training.optimizers import build_optimizer
from face_bias.training.schedulers import build_scheduler
from face_bias.training.trainer import Trainer, build_loss
from face_bias.utils.logging import setup_logging
from face_bias.utils.reproducibility import seed_from_config


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _build_model(config: dict, num_classes: int) -> torch.nn.Module:
    model_cfg = config["model"]
    return LResNet50E_IR(
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        head=model_cfg["head"],
        pretrained=model_cfg.get("pretrained", True),
        arcface_s=model_cfg["arcface_s"],
        arcface_m=model_cfg["arcface_m"],
        arcface_easy_margin=model_cfg["arcface_easy_margin"],
    )


def _maybe_start_mlflow(run_id: str, config: dict, output_dir: Path):
    try:
        import mlflow
    except ImportError:
        logging.info("mlflow not installed - skipping experiment tracking")
        return None

    mlflow.set_tracking_uri((output_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(config["model"]["name"])
    run = mlflow.start_run(run_name=run_id)
    mlflow.log_params(
        {
            "model.head": config["model"]["head"],
            "model.dropout": config["model"]["dropout"],
            "training.optimizer": config["training"]["optimizer"],
            "training.scheduler": config["training"]["scheduler"],
            "training.loss_function": config["training"]["loss_function"],
            "training.learning_rate": config["training"]["learning_rate"],
            "training.batch_size": config["training"]["batch_size"],
            "training.num_epochs": config["training"]["num_epochs"],
            "seed": config["training"]["random_state"],
            "run_id": run_id,
        }
    )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a face recognition model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for checkpoints, logs and MLflow runs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.num_epochs from the config (useful for smoke runs).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.epochs is not None:
        logging.info(f"Overriding num_epochs: {config['training']['num_epochs']} -> {args.epochs}")
        config["training"]["num_epochs"] = args.epochs
    run_id = setup_logging(config, "log_training_file")
    seed_from_config(config)
    device = _resolve_device(args.device)
    logging.info(f"Training run_id={run_id} device={device} config={args.config}")

    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, label_encoder, num_classes = setup_dataset(config)
    class_names = list(label_encoder.classes_)
    logging.info(f"num_classes={num_classes} classes={class_names}")

    model = _build_model(config, num_classes=num_classes)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(
        optimizer,
        config,
        steps_per_epoch=len(dataloaders["train"]),
    )

    mlflow_run = _maybe_start_mlflow(run_id, config, output_dir)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=class_names,
        checkpoint_dir=output_dir / "checkpoints",
        mlflow_run=mlflow_run,
    )

    result = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        epochs=config["training"]["num_epochs"],
    )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    logging.info(f"Training finished. History: {history_path}")
    logging.info(f"Best checkpoint: {result['best_checkpoint']}")

    if mlflow_run is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(result["best_checkpoint"])
            mlflow.end_run()
        except ImportError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
