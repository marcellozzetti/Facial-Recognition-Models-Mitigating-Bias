"""Optimizer factory.

The MBA's experiments toggled between SGD and AdamW; both are kept here
so that the same training entrypoint can reproduce every experiment by
swapping one config value.
"""

from typing import Any

import torch
from torch.optim import SGD, AdamW, Optimizer


def build_optimizer(parameters, config: dict[str, Any]) -> Optimizer:
    """Build an Optimizer from ``config['training']``.

    Recognised values for ``optimizer``:
    - ``"sgd"``: SGD with momentum=0.9, weight_decay=5e-4 (MBA defaults).
    - ``"adamw"``: AdamW with weight_decay=5e-4 (MBA default).
    """
    name = config["training"]["optimizer"].lower()
    lr = float(config["training"]["learning_rate"])

    if name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=5e-4)
    raise ValueError(f"Unknown optimizer {name!r} (expected 'sgd' or 'adamw')")


def parameter_group_count(optimizer: torch.optim.Optimizer) -> int:
    """Convenience wrapper used by smoke tests."""
    return len(optimizer.param_groups)
