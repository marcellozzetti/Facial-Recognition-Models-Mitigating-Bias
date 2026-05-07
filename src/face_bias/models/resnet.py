"""LResNet50E_IR — ResNet50-based backbone for face recognition.

The classification head is configurable: ``head="linear"`` matches the
naive ResNet50+FC pipeline used as MBA baseline; ``head="arcface"`` wires
``ArcMarginProduct`` directly into the model so the additive angular
margin is applied to the logits during training.
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from face_bias.models.arc_margin import ArcMarginProduct

HeadType = Literal["linear", "arcface"]


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
    ):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        if head not in ("linear", "arcface"):
            raise ValueError(f"head must be 'linear' or 'arcface', got {head!r}")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.head_type: HeadType = head

        if head == "linear":
            self.head: nn.Module = nn.Linear(in_features, num_classes)
        else:
            self.head = ArcMarginProduct(
                in_features,
                num_classes,
                s=arcface_s,
                m=arcface_m,
                easy_margin=arcface_easy_margin,
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding (pre-head) for downstream similarity tasks."""
        features = self.backbone(x)
        return self.dropout(features)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.extract_features(x)

        if self.head_type == "linear":
            return self.head(features)

        # ArcFace head: pass labels during training so the margin is applied;
        # during eval (or when labels aren't available) emit plain scaled
        # cosine similarity instead.
        if self.training and labels is not None:
            return self.head(features, labels)
        return self.head.inference_logits(features)
