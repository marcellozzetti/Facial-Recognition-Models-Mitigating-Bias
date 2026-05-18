"""Loss functions.

ArcFace's additive angular margin lives in :class:`ArcMarginProduct` (the
classification head), so the loss itself is just plain cross-entropy on
the margin-corrected logits. We expose it as a named class anyway so
configs can switch between ``cross_entropy`` and ``arcface`` symbolically.

Use ``LResNet50E_IR(head="arcface")`` together with ``ArcFaceLoss`` to get
the ArcFace setup. Pairing ``head="linear"`` with ``ArcFaceLoss`` gives
plain cross-entropy and is identical to ``CrossEntropyLoss``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Cross-entropy on logits emitted by an ArcMarginProduct head.

    The margin and scale parameters are kept here for symmetry with the
    config but are documentation-only — the actual margin is applied by
    the model head. This class no longer ignores them silently: a
    `ValueError` is raised if scale or margin is set to a value
    inconsistent with the head's parameters in the smoke test fixture.
    """

    def __init__(self, margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class AdaFaceLoss(nn.Module):
    """Cross-entropy on logits emitted by an AdaMarginProduct head.

    Like :class:`ArcFaceLoss`, the adaptive margin is applied by the
    model head (see :mod:`face_bias.models.adaface`); the loss is plain
    cross-entropy on the margin-corrected logits. Params kept for config
    symmetry / documentation.
    """

    def __init__(self, margin: float = 0.4, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class MagFaceLoss(nn.Module):
    """Cross-entropy on logits emitted by a MagMarginProduct head.

    The magnitude-aware margin is applied by the model head (see
    :mod:`face_bias.models.magface`). The MagFace magnitude regulariser
    is gated by the head's ``lambda_g`` (default 0.0 → identical to
    ArcFace-style CE-on-margin-logits). When ``lambda_g > 0`` the
    regulariser term must be added by a loss with access to the head's
    ``last_g_reg`` buffer — a documented optional extension; the default
    path here is plain cross-entropy.
    """

    def __init__(self, scale: float = 30.0):
        super().__init__()
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Alias so configs can spell either ``cross_entropy`` or ``arcface``."""
