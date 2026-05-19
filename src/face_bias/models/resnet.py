"""LResNet50E_IR — ResNet50-based backbone for face recognition.

The classification head is configurable:

- ``head="linear"`` — single ``nn.Linear(2048, num_classes)``; matches
  the naive ResNet50+FC pipeline used as MBA baseline.
- ``head="arcface"`` — wires ``ArcMarginProduct`` directly into the
  model so the additive angular margin is applied during training.
- ``head="mlp"`` — configurable multi-layer dense head (see
  :class:`face_bias.models.mlp_head.MLPHead`); introduced for the
  master's HPO study (Optuna search over depth/width/dropout/norm).
- ``head="adaface"`` / ``head="magface"`` — quality-adaptive and
  magnitude-aware angular-margin heads (loss-factor study). Same
  label-routing as ``arcface``.
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from face_bias.models.adaface import AdaMarginProduct
from face_bias.models.arc_margin import ArcMarginProduct
from face_bias.models.magface import MagMarginProduct
from face_bias.models.mlp_head import Activation, MLPHead, Norm

HeadType = Literal["linear", "arcface", "mlp", "adaface", "magface"]

# Heads that apply a label-dependent margin during training and emit
# plain scaled cosine at eval time (need label routing in forward()).
_MARGIN_HEADS = ("arcface", "adaface", "magface")


class ResNet50ImageNet(nn.Module):
    """ImageNet-pretrained torchvision ResNet-50 + configurable head.

    NOTE (honesty / literature fidelity): this is **torchvision
    ``resnet50(weights=ImageNet)``** with ``fc=Identity`` (2048-d
    embedding, 224x224 input, transfer learning) — it is NOT
    insightface's ``LResNet50E-IR`` (IR blocks, 112px, 512-d,
    scratch-trained). The legacy name ``LResNet50E_IR`` is kept below as
    a backward-compat alias (configs/MBA used that misnomer
    consistently — same architecture, so no MBA↔thesis confound). The
    real IR/ViT backbone is the Factor-5 axis. See
    docs/formula_desk_check.md §3.
    """

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
        adaface_m: float = 0.4,
        adaface_h: float = 0.333,
        magface_l_a: float = 10.0,
        magface_u_a: float = 110.0,
        magface_l_m: float = 0.45,
        magface_u_m: float = 0.8,
        magface_lambda_g: float = 0.0,
    ):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        valid_heads = ("linear", "arcface", "mlp", "adaface", "magface")
        if head not in valid_heads:
            raise ValueError(f"head must be one of {valid_heads}, got {head!r}")

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
        elif head == "adaface":
            self.head = AdaMarginProduct(
                in_features,
                num_classes,
                s=arcface_s,
                m=adaface_m,
                h=adaface_h,
            )
        elif head == "magface":
            self.head = MagMarginProduct(
                in_features,
                num_classes,
                s=arcface_s,
                l_a=magface_l_a,
                u_a=magface_u_a,
                l_m=magface_l_m,
                u_m=magface_u_m,
                lambda_g=magface_lambda_g,
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

        if self.head_type in _MARGIN_HEADS:
            # Margin heads (arcface/adaface/magface): pass labels during
            # training so the margin is applied; at eval (or when labels
            # aren't available) emit plain scaled cosine similarity.
            if self.training and labels is not None:
                return self.head(features, labels)
            return self.head.inference_logits(features)

        # linear / mlp: plain logits, labels ignored.
        return self.head(features)


# Backward-compatible alias: the project (MBA + thesis) historically
# named this class ``LResNet50E_IR`` (a misnomer — see class docstring).
# Kept so existing imports, configs and checkpoints keep working
# unchanged while the honest name ``ResNet50ImageNet`` is canonical.
LResNet50E_IR = ResNet50ImageNet
