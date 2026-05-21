"""Optimizer factory.

The MBA's experiments toggled between SGD and AdamW; both are kept here
so that the same training entrypoint can reproduce every experiment by
swapping one config value.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer


def _maybe_split_param_groups(
    model_or_params, config: dict[str, Any], lr: float
):
    """Return either an iterable of params or a list of two param-groups.

    If ``model_or_params`` is an ``nn.Module`` AND ``config['training']
    ['lr_backbone']`` is set, split into two groups: ``model.backbone``
    params at ``lr_backbone``; everything else at ``lr``. Used by the
    FineFACE-recipe anchor (differential LR — backbone fine-tuned
    gently, new head trained aggressively).
    """
    lr_backbone = config["training"].get("lr_backbone")
    if lr_backbone is None or not isinstance(model_or_params, nn.Module):
        if isinstance(model_or_params, nn.Module):
            return model_or_params.parameters()
        return model_or_params

    backbone = getattr(model_or_params, "backbone", None)
    if backbone is None:
        raise ValueError(
            "lr_backbone set but model has no .backbone attribute to split."
        )
    bb_param_ids = {id(p) for p in backbone.parameters()}
    backbone_params = [p for p in model_or_params.parameters() if id(p) in bb_param_ids]
    other_params = [p for p in model_or_params.parameters() if id(p) not in bb_param_ids]
    return [
        {"params": backbone_params, "lr": float(lr_backbone)},
        {"params": other_params, "lr": lr},
    ]


def build_optimizer(model_or_params, config: dict[str, Any]) -> Optimizer:
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
    parameters = _maybe_split_param_groups(model_or_params, config, lr)

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
