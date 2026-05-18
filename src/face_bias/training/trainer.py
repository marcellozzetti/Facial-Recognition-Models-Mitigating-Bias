"""Trainer — train/val loop with MLflow tracking and best-checkpoint saving.

Designed to reproduce the MBA experiment matrix (LResNet50E_IR with
linear or arcface head, ArcFaceLoss or CrossEntropy, SGD or AdamW,
OneCycleLR or CosineAnnealingWarmRestarts) and emit per-class metrics
on every epoch so fairness disparities are visible during training.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from face_bias.evaluation.metrics import classification_metrics, fairness_audit
from face_bias.models.losses import (
    AdaFaceLoss,
    ArcFaceLoss,
    CrossEntropyLoss,
    MagFaceLoss,
)
from face_bias.training.callbacks import EarlyStopping, ModelCheckpoint
from face_bias.training.schedulers import is_step_per_batch


def _save_last_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    """Persist the latest model state regardless of whether it is the best."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "class_names": class_names,
    }
    torch.save(payload, path)


def build_loss(config: dict[str, Any]) -> nn.Module:
    name = config["training"]["loss_function"].lower()
    if name == "cross_entropy":
        return CrossEntropyLoss()
    if name == "arcface":
        return ArcFaceLoss(
            margin=config["model"].get("arcface_m", 0.5),
            scale=config["model"].get("arcface_s", 30.0),
        )
    if name == "adaface":
        return AdaFaceLoss(
            margin=config["model"].get("adaface_m", 0.4),
            scale=config["model"].get("arcface_s", 30.0),
        )
    if name == "magface":
        return MagFaceLoss(scale=config["model"].get("arcface_s", 30.0))
    raise ValueError(f"Unknown loss_function {name!r}")


class Trainer:
    """Plain training loop. Optional MLflow logging if mlflow is installed."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        *,
        class_names: list[str],
        checkpoint_dir: str | Path,
        early_stopping: EarlyStopping | None = None,
        mlflow_run=None,
        grad_clip_norm: float | None = 5.0,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = class_names
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint = ModelCheckpoint(self.checkpoint_dir, mode="min", filename="best.pt")
        self.last_checkpoint_path = self.checkpoint_dir / "last.pt"
        self.early_stopping = early_stopping
        self.mlflow_run = mlflow_run
        # Global gradient-norm clipping. None disables it. Default 5.0 keeps
        # ArcFace + AdamW + dropout=0.5 (Exp 10 in the smoke run) from
        # diverging on the first batch.
        self.grad_clip_norm = grad_clip_norm
        # Mixed precision: opt-in so the 11-experiment replication stays
        # numerically reproducible. New training recipes (e.g. the Pareto
        # refit at 25 epochs) flip ``training.use_amp: true`` in their YAML.
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    # ---- one epoch ----

    def _forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # The ArcFace head needs labels during training; the linear head
        # ignores them. Trainer always passes labels — model decides.
        if hasattr(self.model, "head_type"):
            return self.model(images, labels=labels)
        return self.model(images)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running = 0.0
        seen = 0
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self._forward(images, labels)
                    loss = self.loss_fn(logits, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self._forward(images, labels)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            if is_step_per_batch(self.scheduler):
                self.scheduler.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)
        if not is_step_per_batch(self.scheduler):
            self.scheduler.step()
        return running / max(seen, 1)

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []
        running = 0.0
        seen = 0
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self._forward(images, labels)
                    loss = self.loss_fn(logits, labels)
            else:
                logits = self._forward(images, labels)
                loss = self.loss_fn(logits, labels)
            # Cast logits back to fp32 so the softmax/argmax/numpy chain
            # stays numerically stable regardless of the train-time dtype.
            logits = logits.float()

            running += loss.item() * images.size(0)
            seen += images.size(0)
            proba = torch.softmax(logits, dim=1)
            all_proba.append(proba.cpu().numpy())
            all_preds.append(proba.argmax(dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_proba = np.concatenate(all_proba)

        metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)
        metrics["loss"] = running / max(seen, 1)
        metrics["fairness"] = fairness_audit(y_true, y_pred, self.class_names)
        return metrics

    # ---- public ----

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)
            elapsed = time.time() - start

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_inequity_rate": val_metrics["fairness"]["f1"]["inequity_rate"],
                "elapsed_s": elapsed,
            }
            history.append(entry)
            logging.info(
                "epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                "val_acc={val_accuracy:.4f} val_f1={val_f1_macro:.4f} "
                "val_IR={val_f1_inequity_rate:.3f} t={elapsed_s:.1f}s".format(**entry)
            )

            self._mlflow_log(entry)
            self.checkpoint.step(
                val_metrics["loss"],
                self.model,
                extra={"epoch": epoch, "metrics": entry, "class_names": self.class_names},
            )
            _save_last_checkpoint(
                self.last_checkpoint_path,
                self.model,
                epoch=epoch,
                metrics=entry,
                class_names=self.class_names,
            )

            if self.early_stopping is not None and self.early_stopping.step(val_metrics["loss"]):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "history": history,
            "best_checkpoint": str(self.checkpoint.path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }

    def _mlflow_log(self, entry: dict[str, Any]) -> None:
        if self.mlflow_run is None:
            return
        try:
            import mlflow
        except ImportError:
            return
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=entry["epoch"])
