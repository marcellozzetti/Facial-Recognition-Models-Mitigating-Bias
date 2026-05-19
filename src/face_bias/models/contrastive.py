"""Supervised contrastive paradigm (Factor 4) — canonical SupCon.

Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

Design decision (matched-basis, see docs/sota_pdf_synthesis.md §2.1):
the loss-factor / dataset-factor batches are end-to-end classifiers
(backbone -> head, evaluated by val_f1_macro). To isolate the
*contrastive paradigm* as one factor without confounding the protocol,
Factor 4 is **one-stage joint**: total loss = CE(head logits) +
lambda * SupCon(projected embedding). Same pipeline, same eval head,
same criterion — only the training objective gains a contrastive term.

SupCon is used in its **canonical** form (no fairness modification).
FSCL/FairCL (fairness-modified contrastive) are SOTA *mitigation*
methods cited as related work, NOT the factor arm — exactly as FineFACE
is for the topology axis. The single-view in-batch form is used (one
augmented view per sample, positives = same-label samples in the
batch): this keeps the data/eval pipeline byte-identical to the other
factors. Full two-view SupCon is scoped to the defense program
(PLANO §5), not the qualification factor isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head (SimCLR/SupCon): Linear-BN-ReLU-Linear, then
    L2-normalised. Used only for the contrastive term; the classifier
    head is untouched. Lives in the model so its params are picked up by
    the optimizer (built from ``model.parameters()``)."""

    def __init__(self, in_features: int, hidden: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=1)


class SupConLoss(nn.Module):
    """Supervised contrastive loss, single-view in-batch form.

    ``embeddings``: (B, D) L2-normalised projections. ``labels``: (B,).
    Positives of anchor i = other samples in the batch with the same
    label. Reduces to the canonical SupCon objective with n_views=1
    (Khosla Eq. 2, ``L_out^sup``). Samples with no same-label partner in
    the batch contribute 0 (cannot form a positive pair).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        b = embeddings.size(0)
        z = F.normalize(embeddings, dim=1)  # idempotent guard

        logits = (z @ z.t()) / self.temperature  # (B, B)
        # numerical stability: subtract per-row max (detached)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(b, dtype=torch.bool, device=device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~self_mask  # same class, not self

        exp_logits = torch.exp(logits) * (~self_mask)  # exclude self from denom
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = pos_mask.sum(dim=1)  # |P(i)|
        valid = pos_per_anchor > 0
        if not valid.any():
            # no positive pair anywhere in the batch -> no contrastive signal
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[
            valid
        ]
        return -mean_log_prob_pos.mean()
