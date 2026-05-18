"""MagFace head — magnitude-aware angular margin (Meng et al., CVPR 2021).

MagFace ties the additive angular margin to the feature magnitude: the
larger ``||f||``, the larger the margin, pulling high-magnitude
(high-quality) samples tighter to their class centre. Same head+margin
pattern as :class:`face_bias.models.arc_margin.ArcMarginProduct`, so the
classification loss is plain cross-entropy on the margin-corrected
logits and the head plugs into ``LResNet50E_IR`` unchanged.

Reference: Meng et al., "MagFace: A Universal Representation for Face
Recognition and Quality Assessment", CVPR 2021. We implement the
magnitude→margin schedule ``m(a)`` (linear between ``[l_a,u_a]`` →
``[l_m,u_m]``). The paper's magnitude regulariser ``g(a)`` is exposed
as ``last_g_reg`` (mean over the batch) and gated by ``lambda_g``;
``lambda_g`` defaults to 0.0 so the loss interface stays identical to
ArcFace. Enabling the regulariser (lambda_g>0) is a documented optional
extension and requires a loss that reads ``last_g_reg`` — see
``docs`` / future work. The magnitude-aware margin (the core MagFace
mechanism being attributed in the loss-factor study) is fully active.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagMarginProduct(nn.Module):
    """MagFace magnitude-aware-margin classification head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        l_a: float = 10.0,
        u_a: float = 110.0,
        l_m: float = 0.45,
        u_m: float = 0.8,
        lambda_g: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        if u_a <= l_a:
            raise ValueError(f"u_a must be > l_a; got l_a={l_a}, u_a={u_a}")
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.lambda_g = lambda_g
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Mean magnitude regulariser over the last training batch; a loss
        # may read this when lambda_g > 0 (optional, default off).
        self.register_buffer("last_g_reg", torch.tensor(0.0))

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features), F.normalize(self.weight))

    def inference_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Scaled cosine similarity (no margin) — use at evaluation time."""
        return self._cosine(features) * self.s

    def _m_of_a(self, a: torch.Tensor) -> torch.Tensor:
        slope = (self.u_m - self.l_m) / (self.u_a - self.l_a)
        return slope * (a - self.l_a) + self.l_m

    def _g_of_a(self, a: torch.Tensor) -> torch.Tensor:
        # Magnitude regulariser: 1/u_a^2 * a + 1/a (monotone, convex).
        return (1.0 / (self.u_a ** 2)) * a + 1.0 / a

    def forward(
        self, features: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        cosine = self._cosine(features).clamp(-1 + 1e-7, 1 - 1e-7)

        if label is None:
            return cosine * self.s

        a = torch.norm(features, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m_a = self._m_of_a(a)  # (B, 1) per-sample margin

        if self.training:
            with torch.no_grad():
                self.last_g_reg = self._g_of_a(a).mean()

        cos_m = torch.cos(m_a)
        sin_m = torch.sin(m_a)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(self.eps))
        phi = cosine * cos_m - sine * sin_m  # cos(theta + m(a))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s
