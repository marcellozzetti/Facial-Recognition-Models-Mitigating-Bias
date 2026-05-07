"""ArcMargin layer (additive angular margin) for face recognition."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """Additive angular margin head (Deng et al., ArcFace, CVPR 2019).

    During training (label provided), an additive margin ``m`` is applied to
    the angle between the feature and the target class weight; during
    inference (no label) plain scaled cosine similarity is returned. Both
    are equivalent to ``s * cos(theta + m * y)`` with ``y`` one-hot.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features), F.normalize(self.weight))

    def inference_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Scaled cosine similarity (no margin). Use this at evaluation time."""
        return self._cosine(features) * self.s

    def forward(self, features: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        cosine = self._cosine(features)

        if label is None:
            return cosine * self.s

        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt(1.0 - cosine**2 + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
