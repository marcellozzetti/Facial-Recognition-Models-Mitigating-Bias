"""Regression tests for the AdaFace and MagFace margin heads (Factor 3).

These mirror tests/smoke/test_arcface_integration.py: the margin must
change the logits whenever a label is supplied (training path) and the
eval path must emit plain scaled cosine; the resulting loss must differ
from plain cross-entropy on a linear head; gradients must flow.
"""

import pytest
import torch
import torch.nn as nn

from face_bias.models import (
    AdaFaceLoss,
    AdaMarginProduct,
    CrossEntropyLoss,
    LResNet50E_IR,
    MagFaceLoss,
    MagMarginProduct,
)

NUM_CLASSES = 7
BATCH = 8
EMBED_DIM = 2048


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------- AdaFace ----------


@pytest.mark.unit
def test_adaface_margin_with_label_differs_from_cosine() -> None:
    head = AdaMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.4)
    head.train()
    feats = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    with_margin = head(feats, labels)
    no_margin = head.inference_logits(feats)

    assert with_margin.shape == no_margin.shape == (BATCH, NUM_CLASSES)
    diffs = (with_margin - no_margin).abs().sum(dim=1)
    assert (diffs > 1e-4).all(), "AdaFace margin produced no change in logits"


@pytest.mark.unit
def test_adaface_eval_path_is_plain_cosine() -> None:
    head = AdaMarginProduct(EMBED_DIM, NUM_CLASSES)
    feats = torch.randn(BATCH, EMBED_DIM)
    assert torch.allclose(head(feats, None), head.inference_logits(feats), atol=1e-5)


@pytest.mark.unit
def test_adaface_gradient_flows() -> None:
    head = AdaMarginProduct(EMBED_DIM, NUM_CLASSES)
    head.train()
    feats = torch.randn(BATCH, EMBED_DIM, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss = AdaFaceLoss()(head(feats, labels), labels)
    loss.backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0


@pytest.mark.unit
def test_adaface_norm_quality_modulates_margin() -> None:
    """Higher-norm (higher-quality) features get a different margin than
    low-norm ones — the defining AdaFace behaviour."""
    head = AdaMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.4)
    head.train()
    labels = torch.zeros(BATCH, dtype=torch.long)
    base = torch.randn(BATCH, EMBED_DIM)
    lo = head(base * 0.1, labels)
    hi = head(base * 10.0, labels)
    assert not torch.allclose(lo, hi, atol=1e-3)


# ---------- MagFace ----------


@pytest.mark.unit
def test_magface_margin_with_label_differs_from_cosine() -> None:
    head = MagMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0)
    head.train()
    feats = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    with_margin = head(feats, labels)
    no_margin = head.inference_logits(feats)
    assert with_margin.shape == (BATCH, NUM_CLASSES)
    assert ((with_margin - no_margin).abs().sum(dim=1) > 1e-4).all()


@pytest.mark.unit
def test_magface_margin_increases_with_magnitude() -> None:
    head = MagMarginProduct(EMBED_DIM, NUM_CLASSES)
    a = torch.tensor([[10.0], [110.0]])
    m = head._m_of_a(a)
    assert m[1].item() > m[0].item()  # larger magnitude -> larger margin
    assert pytest.approx(m[0].item(), abs=1e-5) == head.l_m
    assert pytest.approx(m[1].item(), abs=1e-5) == head.u_m


@pytest.mark.unit
def test_magface_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="u_a must be > l_a"):
        MagMarginProduct(EMBED_DIM, NUM_CLASSES, l_a=100.0, u_a=10.0)


@pytest.mark.unit
def test_magface_gradient_flows() -> None:
    head = MagMarginProduct(EMBED_DIM, NUM_CLASSES)
    head.train()
    feats = torch.randn(BATCH, EMBED_DIM, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    MagFaceLoss()(head(feats, labels), labels).backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0


# ---------- integration with LResNet50E_IR ----------


@pytest.mark.unit
@pytest.mark.parametrize("head_name", ["adaface", "magface"])
def test_resnet_margin_head_routes_labels(head_name: str) -> None:
    model = LResNet50E_IR(num_classes=NUM_CLASSES, head=head_name, pretrained=False)
    model.backbone = nn.Identity()  # feed embeddings directly
    feats = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    model.train()
    train_logits = model(feats, labels)
    model.eval()
    eval_logits = model(feats)
    assert train_logits.shape == eval_logits.shape == (BATCH, NUM_CLASSES)
    assert not torch.allclose(train_logits, eval_logits, atol=1e-3), (
        f"{head_name}: training must apply margin; eval must not"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "head_name,loss_cls", [("adaface", AdaFaceLoss), ("magface", MagFaceLoss)]
)
def test_margin_loss_differs_from_linear_ce(head_name, loss_cls) -> None:
    """Same backbone embeddings, margin head vs linear head: the loss
    must differ — proving the margin actually changes training signal."""
    feats = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    linear = nn.Linear(EMBED_DIM, NUM_CLASSES)
    model = LResNet50E_IR(num_classes=NUM_CLASSES, head=head_name, pretrained=False)
    model.backbone = nn.Identity()
    model.dropout = nn.Identity()
    model.train()

    ce = CrossEntropyLoss()(linear(feats), labels)
    margin = loss_cls()(model(feats, labels), labels)
    assert not torch.isclose(ce, margin, atol=1e-3)


@pytest.mark.unit
def test_resnet_rejects_unknown_head_still() -> None:
    with pytest.raises(ValueError, match="head must be one of"):
        LResNet50E_IR(num_classes=NUM_CLASSES, head="cosface", pretrained=False)  # type: ignore[arg-type]
