"""Loss functions.

NOTE: bug B4 from REVIEW_AND_PLAN.md is preserved here during Sprint A
(mechanical migration). The forward currently calls cross_entropy and ignores
the margin parameters; this will be wired up to ArcMarginProduct in Sprint B.
"""

import math

import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Implements the ArcFace loss for face recognition tasks."""

    def __init__(self, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)
