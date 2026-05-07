"""Training callbacks: early stopping and best-model checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when ``metric`` stops improving for ``patience`` epochs."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        improved = (self.mode == "min" and metric < self.best - self.min_delta) or (
            self.mode == "max" and metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Save the model when the monitored metric improves."""

    def __init__(self, dirpath: str | Path, mode: str = "min", filename: str = "best.pt"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.dirpath = Path(dirpath)
        self.mode = mode
        self.filename = filename
        self.best: float | None = None

    @property
    def path(self) -> Path:
        return self.dirpath / self.filename

    def step(self, metric: float, model: nn.Module, *, extra: dict | None = None) -> bool:
        improved = (
            self.best is None
            or (self.mode == "min" and metric < self.best)
            or (self.mode == "max" and metric > self.best)
        )
        if not improved:
            return False

        self.best = metric
        self.dirpath.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "metric": metric,
            **(extra or {}),
        }
        torch.save(payload, self.path)
        logging.info(f"Saved checkpoint metric={metric:.4f} at {self.path}")
        return True
