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
    - ``"sgd"``: SGD with momentum=0.9 (wd from config or 5e-4 default).
    - ``"adamw"``: AdamW (wd from config or 5e-4 default).
    - ``"adam"``: alias of AdamW with wd from config (default 0.0) — used by
      the FairFace-recipe anchor (Kärkkäinen & Joo report "ADAM lr=1e-4"
      without weight decay; cf. docs/baseline_positioning.md).
    """
    name = config["training"]["optimizer"].lower()
    lr = float(config["training"]["learning_rate"])
    cfg_wd = config["training"].get("weight_decay")

    if name == "sgd":
        wd = float(cfg_wd) if cfg_wd is not None else 5e-4
        return SGD(parameters, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adamw":
        wd = float(cfg_wd) if cfg_wd is not None else 5e-4
        return AdamW(parameters, lr=lr, weight_decay=wd)
    if name == "adam":
        # Adam ≡ AdamW with wd=0 (no decoupled decay); default 0.0 matches
        # the FairFace paper's unspecified weight decay.
        wd = float(cfg_wd) if cfg_wd is not None else 0.0
        return AdamW(parameters, lr=lr, weight_decay=wd)
    raise ValueError(
        f"Unknown optimizer {name!r} (expected 'sgd', 'adamw', or 'adam')"
    )


def parameter_group_count(optimizer: torch.optim.Optimizer) -> int:
    """Convenience wrapper used by smoke tests."""
    return len(optimizer.param_groups)
