"""Unit tests para src/face_bias/baselines/*.py — Etapa 4."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from face_bias.baselines import (
    AdversarialDebiasLoss,
    AdversarialHead,
    CrossLayerMutualAttention,
    FSCLPlusLoss,
    GroupDROLoss,
    gradient_reverse,
)


# ---------------------------------------------------------------------------
# FSCLPlusLoss
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fscl_plus_forward_returns_scalar():
    torch.manual_seed(0)
    z = torch.randn(16, 128)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.randint(0, 7, (16,))
    a = torch.randint(0, 2, (16,))
    loss = FSCLPlusLoss()(z, y, a)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.unit
def test_fscl_plus_gradient_flows():
    z = torch.randn(8, 64, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    a = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = FSCLPlusLoss()(z, y, a)
    loss.backward()
    assert z.grad is not None
    assert torch.any(z.grad != 0)


@pytest.mark.unit
def test_fscl_plus_handles_single_sample():
    z = torch.randn(1, 32)
    loss = FSCLPlusLoss()(z, torch.tensor([0]), torch.tensor([0]))
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# GroupDROLoss
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_group_dro_updates_weights():
    torch.manual_seed(0)
    loss_fn = GroupDROLoss(n_groups=4, eta=0.1)
    logits = torch.randn(32, 7)
    labels = torch.randint(0, 7, (32,))
    # grupo 2 sempre pega loss maior
    groups = torch.randint(0, 4, (32,))
    initial_weights = loss_fn.group_weights.clone()
    for _ in range(3):
        loss_fn(logits, labels, groups)
    # pesos devem ter mudado
    assert not torch.allclose(loss_fn.group_weights, initial_weights)
    # pesos sempre somam 1
    assert loss_fn.group_weights.sum().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_group_dro_handles_missing_group():
    loss_fn = GroupDROLoss(n_groups=4)
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))
    groups = torch.zeros(8, dtype=torch.long)  # todos no grupo 0
    loss = loss_fn(logits, labels, groups)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# AdversarialDebiasLoss + GradientReverse
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gradient_reverse_inverts_sign():
    x = torch.tensor([2.0], requires_grad=True)
    y = gradient_reverse(x, lambda_=1.5)
    y.sum().backward()
    assert x.grad.item() == pytest.approx(-1.5)


@pytest.mark.unit
def test_adversarial_head_shape():
    head = AdversarialHead(in_dim=64, n_sensitive=2)
    out = head(torch.randn(3, 64))
    assert out.shape == (3, 2)


@pytest.mark.unit
def test_adversarial_debias_loss_scalar():
    loss_fn = AdversarialDebiasLoss(lambda_adv=0.5)
    logits_main = torch.randn(4, 7)
    y_main = torch.tensor([0, 1, 2, 3])
    logits_adv = torch.randn(4, 2)
    y_sens = torch.tensor([0, 1, 0, 1])
    loss = loss_fn(logits_main, y_main, logits_adv, y_sens)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# CrossLayerMutualAttention (FineFACE)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cross_layer_attention_preserves_shapes():
    attn = CrossLayerMutualAttention(dim_low=96, dim_high=384, n_heads=4)
    low = torch.randn(2, 96, 8, 8)
    high = torch.randn(2, 384, 4, 4)
    out_low, out_high = attn(low, high)
    assert out_low.shape == low.shape
    assert out_high.shape == high.shape


@pytest.mark.unit
def test_cross_layer_attention_gradient_flows():
    attn = CrossLayerMutualAttention(dim_low=32, dim_high=64)
    low = torch.randn(1, 32, 4, 4, requires_grad=True)
    high = torch.randn(1, 64, 2, 2, requires_grad=True)
    out_low, out_high = attn(low, high)
    (out_low.sum() + out_high.sum()).backward()
    assert low.grad is not None
    assert high.grad is not None
