"""AdaFace head — quality-adaptive angular margin (Kim et al., CVPR 2022).

AdaFace makes the additive angular margin a function of the feature
norm, used as a proxy for image quality: low-norm (low-quality) samples
get a *de-emphasising* margin, high-norm (high-quality) samples get an
*emphasising* one. This is the same head+margin pattern as
:class:`face_bias.models.arc_margin.ArcMarginProduct` — the loss is
plain cross-entropy on the margin-corrected logits — so it plugs into
``LResNet50E_IR`` and the trainer with no interface change.

Reference: Kim, Jain & Liu, "AdaFace: Quality Adaptive Margin for Face
Recognition", CVPR 2022. We implement the batch-statistics feature-norm
normalisation with EMA running buffers (paper momentum 0.99), the
angular term ``g_angle = -m * z`` and the additive term
``g_add = m * z + m`` with ``z = clip((||f|| - mu) / (sigma/h), -1, 1)``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaMarginProduct(nn.Module):
    """AdaFace adaptive-margin classification head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.4,
        h: float = 0.333,
        ema_momentum: float = 0.99,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.h = h
        self.ema_momentum = ema_momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Running batch statistics of the feature norm (EMA, like the
        # paper). Init near THIS backbone's measured regime (probe:
        # ||f|| ~ 9 ± 2 for ImageNet ResNet-50 @224 on FairFace) instead
        # of the paper's scratch-IR-scale defaults (20/100). With
        # momentum 0.99 a far-off init keeps ẑ≈0 (margin ≈ constant) for
        # hundreds of batches, delaying AdaFace's adaptive behaviour —
        # see docs/formula_desk_check.md §2.2.
        self.register_buffer("batch_mean", torch.tensor(9.0))
        self.register_buffer("batch_std", torch.tensor(2.0))

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features), F.normalize(self.weight))

    def inference_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Scaled cosine similarity (no margin) — use at evaluation time."""
        return self._cosine(features) * self.s

    def forward(
        self, features: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        cosine = self._cosine(features).clamp(-1 + 1e-7, 1 - 1e-7)

        if label is None:
            return cosine * self.s

        # Feature norm as a quality proxy; detach so the norm pathway does
        # not backprop through the margin (paper treats it as a statistic).
        safe_norm = torch.norm(features, dim=1, keepdim=True).clamp(self.eps, 100.0)
        norm = safe_norm.detach()

        if self.training:
            mean = norm.mean()
            std = norm.std().clamp_min(self.eps)
            with torch.no_grad():
                self.batch_mean.lerp_(mean, 1 - self.ema_momentum)
                self.batch_std.lerp_(std, 1 - self.ema_momentum)

        z = (norm - self.batch_mean) / (self.batch_std / self.h + self.eps)
        z = z.clamp(-1.0, 1.0)  # (B, 1)

        g_angle = -self.m * z          # angular margin term
        g_add = self.m * z + self.m    # additive margin term

        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Target class: cos(theta + g_angle) - g_add ; others: cos(theta).
        target = torch.cos(theta + g_angle) - g_add
        output = one_hot * target + (1.0 - one_hot) * cosine
        return output * self.s
