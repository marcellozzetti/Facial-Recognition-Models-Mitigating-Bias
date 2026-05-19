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
from face_bias.training.callbacks import EarlyStopping
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
        mlp_hidden_dims=model_cfg.get("mlp_hidden_dims", [512]),
        mlp_activation=model_cfg.get("mlp_activation", "relu"),
        mlp_dropout=model_cfg.get("mlp_dropout", 0.3),
        mlp_norm=model_cfg.get("mlp_norm", "none"),
        adaface_m=model_cfg.get("adaface_m", 0.4),
        adaface_h=model_cfg.get("adaface_h", 0.333),
        magface_l_a=model_cfg.get("magface_l_a", 10.0),
        magface_u_a=model_cfg.get("magface_u_a", 110.0),
        magface_l_m=model_cfg.get("magface_l_m", 0.45),
        magface_u_m=model_cfg.get("magface_u_m", 0.8),
        magface_lambda_g=model_cfg.get("magface_lambda_g", 0.0),
    )


def _maybe_start_mlflow(run_id: str, config: dict, mlruns_root: Path):
    try:
        import mlflow
    except ImportError:
        logging.info("mlflow not installed - skipping experiment tracking")
        return None

    # Centralised mlruns folder under the top-level output dir so the
    # absolute path stays within Windows' MAX_PATH (260 chars) even when
    # the run-specific subdir is several levels deep (e.g. when invoked
    # by run_all_experiments.py).
    mlruns_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_root.resolve().as_uri())
    mlflow.set_experiment(config["model"]["name"])
    run = mlflow.start_run(run_name=run_id)
    params = {
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
    if config["model"]["head"] == "mlp":
        params.update(
            {
                "model.mlp_hidden_dims": str(config["model"].get("mlp_hidden_dims", [512])),
                "model.mlp_activation": config["model"].get("mlp_activation", "relu"),
                "model.mlp_dropout": config["model"].get("mlp_dropout", 0.3),
                "model.mlp_norm": config["model"].get("mlp_norm", "none"),
            }
        )
    mlflow.log_params(params)
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

    output_root = Path(args.output_dir)
    output_dir = output_root / run_id
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

    mlflow_run = _maybe_start_mlflow(run_id, config, output_root / "mlruns")

    monitor = config["training"].get("checkpoint_metric", "val_f1_macro")
    monitor_mode = "min" if monitor == "val_loss" else "max"
    patience = config["training"].get("early_stopping_patience")
    early_stopping = (
        EarlyStopping(patience=patience, mode=monitor_mode) if patience else None
    )
    if early_stopping is not None:
        logging.info(
            f"EarlyStopping enabled (mode={monitor_mode}, patience={patience} "
            f"epochs on {monitor})"
        )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=class_names,
        checkpoint_dir=output_dir / "checkpoints",
        mlflow_run=mlflow_run,
        early_stopping=early_stopping,
        grad_clip_norm=config["training"].get("grad_clip_norm"),
        use_amp=config["training"].get("use_amp", False),
        monitor=monitor,
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
