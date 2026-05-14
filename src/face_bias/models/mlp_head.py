"""MLP classification head for face-recognition backbones.

The MBA baseline plugs a single ``nn.Linear(2048, num_classes)`` on top of
ResNet50's pooled features. The orientador (kickoff meeting 2026-05-11,
diretriz nº 2) asked for a configurable multi-layer head with dense
hidden layers so Optuna can search the head topology while the backbone
stays fixed. This module implements that head; the geometry (depth,
width, activation, dropout, normalisation) is driven entirely by the
constructor args so the search space is declarative.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Activation = Literal["relu", "gelu", "silu", "tanh"]
Norm = Literal["none", "batchnorm", "layernorm"]


def _build_activation(name: Activation) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}")


def _build_norm(name: Norm, num_features: int) -> nn.Module | None:
    if name == "none":
        return None
    if name == "batchnorm":
        return nn.BatchNorm1d(num_features)
    if name == "layernorm":
        return nn.LayerNorm(num_features)
    raise ValueError(f"Unknown norm {name!r}")


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head.

    Topology: ``[Linear -> (Norm) -> Activation -> (Dropout)] * depth``
    followed by a final ``Linear(prev, num_classes)`` that emits raw
    logits (no activation, no normalisation).

    Parameters
    ----------
    in_features:
        Embedding dimension produced by the backbone (2048 for ResNet50).
    num_classes:
        Number of output classes.
    hidden_dims:
        Width of each hidden layer. ``[]`` recovers the linear baseline
        (which is what the legacy ``head="linear"`` path provides), so by
        convention ``MLPHead`` is only instantiated with at least one
        hidden layer.
    activation, dropout, norm:
        Applied identically to every hidden block.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: list[int],
        activation: Activation = "relu",
        dropout: float = 0.3,
        norm: Norm = "none",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLPHead requires at least one hidden layer; got hidden_dims=[]")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            normed = _build_norm(norm, h)
            if normed is not None:
                layers.append(normed)
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.out_features = num_classes
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
