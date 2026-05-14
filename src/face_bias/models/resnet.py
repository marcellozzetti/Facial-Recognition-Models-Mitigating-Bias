"""LResNet50E_IR — ResNet50-based backbone for face recognition.

The classification head is configurable:

- ``head="linear"`` — single ``nn.Linear(2048, num_classes)``; matches
  the naive ResNet50+FC pipeline used as MBA baseline.
- ``head="arcface"`` — wires ``ArcMarginProduct`` directly into the
  model so the additive angular margin is applied during training.
- ``head="mlp"`` — configurable multi-layer dense head (see
  :class:`face_bias.models.mlp_head.MLPHead`); introduced for the
  master's HPO study (Optuna search over depth/width/dropout/norm).
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from face_bias.models.arc_margin import ArcMarginProduct
from face_bias.models.mlp_head import Activation, MLPHead, Norm

HeadType = Literal["linear", "arcface", "mlp"]


class LResNet50E_IR(nn.Module):
    """ResNet50-based model tailored for facial recognition tasks."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.5,
        head: HeadType = "linear",
        pretrained: bool = True,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        arcface_easy_margin: bool = False,
        mlp_hidden_dims: list[int] | None = None,
        mlp_activation: Activation = "relu",
        mlp_dropout: float = 0.3,
        mlp_norm: Norm = "none",
    ):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        if head not in ("linear", "arcface", "mlp"):
            raise ValueError(f"head must be 'linear', 'arcface' or 'mlp', got {head!r}")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.head_type: HeadType = head

        if head == "linear":
            self.head: nn.Module = nn.Linear(in_features, num_classes)
        elif head == "arcface":
            self.head = ArcMarginProduct(
                in_features,
                num_classes,
                s=arcface_s,
                m=arcface_m,
                easy_margin=arcface_easy_margin,
            )
        else:  # head == "mlp"
            hidden = mlp_hidden_dims if mlp_hidden_dims else [512]
            self.head = MLPHead(
                in_features=in_features,
                num_classes=num_classes,
                hidden_dims=hidden,
                activation=mlp_activation,
                dropout=mlp_dropout,
                norm=mlp_norm,
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding (pre-head) for downstream similarity tasks."""
        features = self.backbone(x)
        return self.dropout(features)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.extract_features(x)

        if self.head_type == "arcface":
            # ArcFace head: pass labels during training so the margin is
            # applied; during eval (or when labels aren't available) emit
            # plain scaled cosine similarity instead.
            if self.training and labels is not None:
                return self.head(features, labels)
            return self.head.inference_logits(features)

        # linear / mlp: plain logits, labels ignored.
        return self.head(features)
