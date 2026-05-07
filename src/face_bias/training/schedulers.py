"""LR scheduler factory."""

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
)


def build_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
    *,
    steps_per_epoch: int,
) -> LRScheduler:
    """Build an LR scheduler from ``config['training']``.

    Recognised values for ``scheduler``:
    - ``"onecyclelr"``: OneCycleLR with max_lr=10*learning_rate over the
      full training horizon (epochs * steps_per_epoch).
    - ``"cosineannealingwarmrestarts"``: T_0=8 epochs, T_mult=1, eta_min=1e-6.
    """
    name = config["training"]["scheduler"].lower()
    lr = float(config["training"]["learning_rate"])
    epochs = int(config["training"]["num_epochs"])

    if name == "onecyclelr":
        return OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    if name == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)
    raise ValueError(
        f"Unknown scheduler {name!r} (expected 'onecyclelr' or 'cosineannealingwarmrestarts')"
    )


def is_step_per_batch(scheduler: LRScheduler) -> bool:
    """Whether ``scheduler.step()`` should be called every batch (vs. epoch)."""
    return isinstance(scheduler, OneCycleLR)
