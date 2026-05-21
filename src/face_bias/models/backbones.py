"""Backbone factory for Factor 5 (backbone axis).

Returns ``(backbone_module, embed_dim)`` with the final classifier
stripped to ``nn.Identity()`` so ``backbone(x)`` emits the pooled
feature vector consumed by the classification head. All backbones share
the same 224x224 RGB ImageNet-normalised input pipeline, matched with
the dataset/loss/topology factors. ImageNet pretrained weights.

- ``resnet50``: torchvision ResNet-50, 2048-d. **The MBA and definitive
  baselines (all current results) use this. Byte-identical to the
  prior hardcoded construction — Factor-5 default preserves
  reproducibility for every existing config/checkpoint.**
- ``vit_b_16``: torchvision ViT-B/16, 768-d (CLS-token after encoder norm).
- ``convnext_tiny``: torchvision ConvNeXt-T, 768-d (GAP'd feature).

Insightface's IR-real (LResNet50E-IR; 112px, 512-d, scratch-trained) is
scoped to the defense program (PLANO §5), not the qualification factor
isolation — see docs/sota_pdf_synthesis.md / formula_desk_check.md.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)

BackboneArch = Literal["resnet50", "resnet34", "vit_b_16", "convnext_tiny"]


def build_backbone(arch: BackboneArch, pretrained: bool) -> tuple[nn.Module, int]:
    """Build the backbone and return ``(module, embed_dim)``.

    The returned ``module`` accepts ``(B, 3, 224, 224)`` and emits
    ``(B, embed_dim)``. The original final classifier is replaced by
    ``nn.Identity()`` so the downstream head (linear/arcface/mlp/…) is
    fed pooled features directly.
    """
    if arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        bb = models.resnet50(weights=weights)
        embed_dim = bb.fc.in_features  # 2048
        bb.fc = nn.Identity()
        return bb, embed_dim

    if arch == "resnet34":
        # FairFace-original recipe anchor (Kärkkäinen & Joo, WACV 2021):
        # the dataset paper used ResNet-34 + ADAM lr=1e-4. Used here as
        # baseline-positioning anchor, NOT as one of the 5 attribution
        # factors. See docs/baseline_positioning.md.
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        bb = models.resnet34(weights=weights)
        embed_dim = bb.fc.in_features  # 512
        bb.fc = nn.Identity()
        return bb, embed_dim

    if arch == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        bb = models.vit_b_16(weights=weights)
        # ViT head is `bb.heads = Sequential(head=Linear(768, num_classes))`.
        # Replacing the whole `heads` with Identity returns the 768-d CLS
        # token after the encoder LayerNorm.
        embed_dim = bb.heads.head.in_features  # 768
        bb.heads = nn.Identity()
        return bb, embed_dim

    if arch == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        bb = models.convnext_tiny(weights=weights)
        # ConvNeXt: classifier = Sequential(LayerNorm2d, Flatten, Linear).
        # Drop the final Linear -> output is the 768-d post-Flatten feature.
        embed_dim = bb.classifier[-1].in_features  # 768
        bb.classifier[-1] = nn.Identity()
        return bb, embed_dim

    raise ValueError(
        f"backbone_arch must be one of resnet50, vit_b_16, convnext_tiny; got {arch!r}"
    )
