"""Group DRO — Distributionally Robust Optimization (Sagawa et al. 2020).

Cap. 4 §Baselines. Substitui minimização do erro médio (ERM) pela
minimização do erro do PIOR GRUPO demográfico:

    L_GroupDRO(θ) = max_g E[ℓ(f_θ(x), y) | grupo=g]

Implementado com pesos por grupo atualizados exponencialmente:

    q_g ← q_g * exp(η * loss_g)      (renormalizado)
    L = Σ_g q_g * loss_g
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupDROLoss(nn.Module):
    """Group DRO wrapping cross-entropy padrão.

    Parâmetros
    ----------
    n_groups:
        Número de grupos demográficos (raças, subgrupos race×gender...).
    eta:
        Taxa de aprendizado dos pesos por grupo (Sagawa usa 0.01–0.1).
    """

    def __init__(self, n_groups: int, eta: float = 0.01):
        super().__init__()
        self.n_groups = int(n_groups)
        self.eta = float(eta)
        # pesos por grupo, atualizados a cada batch (não são parâmetros treináveis)
        self.register_buffer("group_weights", torch.ones(self.n_groups) / self.n_groups)

    def forward(
        self,
        logits: torch.Tensor,   # (N, C)
        labels: torch.Tensor,   # (N,)
        groups: torch.Tensor,   # (N,) valores em [0, n_groups)
    ) -> torch.Tensor:
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        # loss média por grupo (0 se grupo ausente no batch)
        group_losses = torch.zeros(self.n_groups, device=logits.device)
        for g in range(self.n_groups):
            mask = groups == g
            if mask.any():
                group_losses[g] = per_sample[mask].mean()

        with torch.no_grad():
            new_w = self.group_weights * torch.exp(self.eta * group_losses.detach())
            new_w = new_w / new_w.sum()
            self.group_weights.copy_(new_w)

        return (self.group_weights * group_losses).sum()
