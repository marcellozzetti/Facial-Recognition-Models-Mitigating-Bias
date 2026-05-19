"""Regression tests for Factor 4 — canonical SupCon (one-stage joint).

Locks: SupCon correctness (same-class pulled together vs different-class
apart), single-view in-batch behaviour, no-positive safety, gradient
flow, projection head shape/normalisation, and the model/trainer wiring
(projection only when enabled; contrastive term differentiable and
changes the loss vs pure CE).
"""

import pytest
import torch
import torch.nn as nn

from face_bias.models import (
    LResNet50E_IR,
    ProjectionHead,
    ResNet50ImageNet,
    SupConLoss,
)

NUM_CLASSES = 7
BATCH = 16
EMBED_DIM = 2048


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------- SupConLoss ----------


@pytest.mark.unit
def test_supcon_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        SupConLoss(temperature=0.0)


@pytest.mark.unit
def test_supcon_lower_when_classes_well_separated() -> None:
    """Embeddings clustered by class must yield a much lower SupCon than
    random embeddings — the defining behaviour."""
    labels = torch.arange(NUM_CLASSES).repeat(3)  # 3 per class, 21 samples
    centers = torch.randn(NUM_CLASSES, 32)
    clustered = centers[labels] + 0.01 * torch.randn(labels.size(0), 32)
    random = torch.randn(labels.size(0), 32)

    loss = SupConLoss(temperature=0.07)
    assert loss(clustered, labels) < loss(random, labels) - 0.5


@pytest.mark.unit
def test_supcon_no_positive_pair_is_safe_zero() -> None:
    """Every label unique -> no positive pair -> 0 contribution, no NaN,
    gradient defined."""
    emb = torch.randn(NUM_CLASSES, 16, requires_grad=True)
    labels = torch.arange(NUM_CLASSES)
    out = SupConLoss()(emb, labels)
    assert torch.isfinite(out) and float(out) == 0.0
    out.backward()  # must not raise


@pytest.mark.unit
def test_supcon_gradient_flows() -> None:
    emb = torch.randn(BATCH, 32, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    SupConLoss()(emb, labels).backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0


# ---------- ProjectionHead ----------


@pytest.mark.unit
def test_projection_head_shape_and_l2_norm() -> None:
    head = ProjectionHead(EMBED_DIM, hidden=256, proj_dim=64)
    head.eval()
    z = head(torch.randn(BATCH, EMBED_DIM))
    assert z.shape == (BATCH, 64)
    assert torch.allclose(z.norm(dim=1), torch.ones(BATCH), atol=1e-5)


# ---------- model wiring ----------


@pytest.mark.unit
def test_resnet_no_projection_by_default() -> None:
    m = LResNet50E_IR(num_classes=NUM_CLASSES, head="linear", pretrained=False)
    assert m.projection is None
    with pytest.raises(RuntimeError, match="projection head not built"):
        m.project(torch.randn(2, EMBED_DIM))


@pytest.mark.unit
def test_resnet_projection_built_when_requested() -> None:
    m = ResNet50ImageNet(
        num_classes=NUM_CLASSES,
        head="linear",
        pretrained=False,
        contrastive_proj_dim=128,
    )
    m.eval()
    feats = torch.randn(BATCH, EMBED_DIM)
    z = m.project(feats)
    assert z.shape == (BATCH, 128)
    assert torch.allclose(z.norm(dim=1), torch.ones(BATCH), atol=1e-5)
    # projection params are part of the model -> picked up by optimizer
    assert any("projection" in n for n, _ in m.named_parameters())


@pytest.mark.unit
def test_joint_loss_differs_from_pure_ce() -> None:
    """CE + lambda*SupCon must differ from CE alone (same logits/feats),
    proving the contrastive term actually changes the training signal."""
    torch.manual_seed(0)
    m = ResNet50ImageNet(
        num_classes=NUM_CLASSES, head="linear", pretrained=False,
        contrastive_proj_dim=64,
    )
    m.backbone = nn.Identity()
    m.train()
    x = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    feats = m.extract_features(x)
    ce = nn.CrossEntropyLoss()(m.head(feats), labels)
    joint = ce + 0.5 * SupConLoss()(m.project(feats), labels)
    assert not torch.isclose(ce, joint)
    joint.backward()  # differentiable end-to-end
