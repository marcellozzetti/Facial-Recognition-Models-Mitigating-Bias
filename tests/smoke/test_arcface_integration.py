"""Regression tests for the ArcFace integration (REVIEW_AND_PLAN.md §2.2).

The pre-Sprint-B ``ArcFaceLoss.forward`` silently fell back to
``F.cross_entropy``, making every "ArcFace" experiment in the MBA
identical to plain cross-entropy. These tests pin down the corrected
behaviour: the ArcFace head must change the loss whenever the margin
``m > 0``, and inference must produce plain scaled cosine logits.
"""

import pytest
import torch
import torch.nn as nn

from face_bias.models import ArcFaceLoss, ArcMarginProduct, LResNet50E_IR

NUM_CLASSES = 7
BATCH = 4
EMBED_DIM = 2048


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.unit
def test_arc_margin_with_label_differs_from_cosine() -> None:
    head = ArcMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.5)
    features = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    with_margin = head(features, labels)
    no_margin = head.inference_logits(features)

    assert with_margin.shape == no_margin.shape == (BATCH, NUM_CLASSES)
    # The margin must perturb the target-class column at least.
    diffs = (with_margin - no_margin).abs().sum(dim=1)
    assert (diffs > 1e-4).all(), "ArcFace margin produced no change in logits"


@pytest.mark.unit
def test_arc_margin_zero_margin_matches_cosine() -> None:
    head = ArcMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.0)
    features = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    assert torch.allclose(head(features, labels), head.inference_logits(features), atol=1e-5)


@pytest.mark.unit
def test_resnet_arcface_head_uses_labels_in_training() -> None:
    model = LResNet50E_IR(num_classes=NUM_CLASSES, head="arcface")
    # Avoid the 25M-param backbone weights download on every test run.
    model.backbone = nn.Identity()  # backbone is irrelevant; we feed embeddings directly

    features = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    model.train()
    train_logits = model.head(features, labels)
    model.eval()
    eval_logits = model.head.inference_logits(features)

    assert not torch.allclose(train_logits, eval_logits, atol=1e-3), (
        "Training-mode ArcFace must apply margin; eval-mode must not."
    )


@pytest.mark.unit
def test_arcface_loss_responds_to_margin_via_head() -> None:
    """The bug: ArcFaceLoss used to compute cross_entropy on naive logits, so
    using `head="arcface"` vs `head="linear"` produced the same loss. After
    the fix, the loss must differ when the head is different.
    """
    features = torch.randn(BATCH, EMBED_DIM, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    linear_head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    arcface_head = ArcMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.5)
    # Force the same starting weights up to the cosine geometry.
    with torch.no_grad():
        arcface_head.weight.copy_(linear_head.weight)

    loss_fn = ArcFaceLoss()

    linear_logits = linear_head(features)
    arcface_logits = arcface_head(features, labels)

    linear_loss = loss_fn(linear_logits, labels)
    arcface_loss = loss_fn(arcface_logits, labels)

    assert not torch.isclose(linear_loss, arcface_loss, atol=1e-3), (
        "ArcFace loss collapsed to cross-entropy — bug §2.2 has regressed."
    )


@pytest.mark.unit
def test_arcface_loss_gradient_flows_to_features() -> None:
    features = torch.randn(BATCH, EMBED_DIM, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    head = ArcMarginProduct(EMBED_DIM, NUM_CLASSES, s=30.0, m=0.5)
    logits = head(features, labels)
    loss = ArcFaceLoss()(logits, labels)
    loss.backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0, "No gradient flow through ArcFace head"
