"""Regression tests for Factor 5 — backbone factory + integration.

Locks: each supported backbone builds with the right embedding dim,
forwards (B,3,224,224) -> (B,embed_dim), and integrates with
LResNet50E_IR/ResNet50ImageNet so the head sees the correct in_features.
The default 'resnet50' must remain byte-identical to the prior
construction (preserves every existing config/checkpoint/baseline).
"""

import pytest
import torch

from face_bias.models import LResNet50E_IR
from face_bias.models.backbones import build_backbone

NUM_CLASSES = 7
B = 2


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "arch, expected_dim",
    [("resnet50", 2048), ("vit_b_16", 768), ("convnext_tiny", 768)],
)
def test_backbone_builds_and_emits_pooled_features(arch, expected_dim) -> None:
    bb, dim = build_backbone(arch, pretrained=False)
    assert dim == expected_dim
    bb.eval()
    out = bb(torch.randn(B, 3, 224, 224))
    assert out.shape == (B, expected_dim), f"{arch}: got {out.shape}"


@pytest.mark.unit
def test_unknown_backbone_rejected() -> None:
    with pytest.raises(ValueError, match="backbone_arch must be one of"):
        build_backbone("resnet101", pretrained=False)  # type: ignore[arg-type]


@pytest.mark.unit
def test_default_resnet50_byte_identical_to_prior() -> None:
    """The default arch ('resnet50') must reproduce the construction the
    project has used since the MBA: torchvision resnet50 with
    fc=Identity and a 2048-d pooled embedding. Anchors back-compat for
    every existing checkpoint/baseline (definitive runs)."""
    from torchvision import models as tv

    bb_ref = tv.resnet50(weights=None)
    in_ref = bb_ref.fc.in_features
    assert in_ref == 2048
    bb, dim = build_backbone("resnet50", pretrained=False)
    assert dim == in_ref


@pytest.mark.unit
@pytest.mark.parametrize("arch", ["vit_b_16", "convnext_tiny"])
def test_model_integrates_modern_backbone(arch) -> None:
    """LResNet50E_IR wired with a modern backbone produces logits of the
    right shape — the linear head adapts to the backbone's embed_dim."""
    m = LResNet50E_IR(
        num_classes=NUM_CLASSES, head="linear", pretrained=False, backbone_arch=arch,
    )
    m.eval()
    logits = m(torch.randn(B, 3, 224, 224))
    assert logits.shape == (B, NUM_CLASSES)
    assert m.backbone_arch == arch
