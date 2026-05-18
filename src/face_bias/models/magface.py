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
``[l_m,u_m]``) AND the magnitude regulariser ``g(a)``, exposed
differentiably as ``last_g_reg`` (mean over the batch) and weighted by
``lambda_g``; the trainer adds ``lambda_g * last_g_reg`` to the loss.

The regulariser is NOT optional in practice: MagFace's mechanism is
inseparable from it. Empirically (see ``scripts/diag_magface_*``), with
``lambda_g=0`` and an ImageNet-pretrained ResNet-50 whose feature norms
fall below ``l_a``, nothing pins ``||f||`` — the norm and representation
collapse and the eval head degenerates to a uniform output. ``g(a)`` is
therefore computed on the *raw, unclamped* norm so its gradient stays
alive below ``l_a`` (the regime that collapsed); the clamped ``a`` is
used only for the margin-schedule domain. Canonical MagFace runs set
``lambda_g>0``.
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

        # Mean magnitude regulariser over the last training batch,
        # DIFFERENTIABLE. The trainer adds ``lambda_g * last_g_reg`` to
        # the loss (canonical MagFace). Plain attribute, not a buffer:
        # it must carry the autograd graph, and it is recomputed every
        # training forward, so it is never persisted nor moved by .to().
        self.last_g_reg: torch.Tensor = torch.tensor(0.0)

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

        raw_norm = torch.norm(features, dim=1, keepdim=True)
        # The MARGIN treats the magnitude as a *statistic*: detach so it
        # does not backprop through the norm. Leaving it attached gives
        # the optimiser a degenerate escape — game ||f|| to shrink the
        # margin instead of learning angles — which collapses the
        # representation (real diagnosis; AdaFace detaches for exactly
        # this reason, and it is the only norm-handling difference
        # between the head that works and this one). The *intended*
        # differentiable magnitude objective is solely lambda_g * g(a),
        # computed below on the un-detached norm — the paper's design.
        a = raw_norm.detach().clamp(self.l_a, self.u_a)  # margin domain
        m_a = self._m_of_a(a)  # (B, 1) per-sample margin

        if self.training:
            # MagFace magnitude regulariser, on the RAW (unclamped) norm
            # and DIFFERENTIABLE. g(a)=1/u_a^2*a + 1/a is convex with
            # minimum at a=u_a, so -grad pulls ||f|| up. Computing it on
            # the unclamped norm keeps the gradient alive *below* l_a —
            # exactly the regime where the norm/representation collapsed
            # when this term was disabled (lambda_g=0). The trainer adds
            # lambda_g * last_g_reg to the loss.
            self.last_g_reg = self._g_of_a(raw_norm.clamp_min(self.eps)).mean()

        cos_m = torch.cos(m_a)
        sin_m = torch.sin(m_a)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(self.eps))
        phi = cosine * cos_m - sine * sin_m  # cos(theta + m(a))

        # Monotonicity guard (ArcFace boundary correction), applied
        # per-sample because the MagFace margin m(a) varies per sample.
        # When theta + m(a) > pi the raw cos(theta+m) stops decreasing
        # and turns back up, so a worse angle would yield a *higher*
        # target logit — inverting the gradient and collapsing the
        # embedding. Replace that region with the monotone linear
        # extension cosine - mm. Using cos(pi-x)=-cos(x),
        # sin(pi-x)=sin(x). Matches the official MagFace/ArcFace
        # reference (easy_margin=False branch).
        th = -cos_m  # cos(pi - m(a))
        mm = sin_m * m_a  # sin(pi - m(a)) * m(a)
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s
