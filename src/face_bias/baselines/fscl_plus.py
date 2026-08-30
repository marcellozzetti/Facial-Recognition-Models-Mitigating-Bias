"""FSCL+ — Fair Supervised Contrastive Learning (Park et al. 2022).

Cap. 4 §Baselines. Extensão do SupCon (Khosla 2020) com controle
explícito de viés: empurra amostras da mesma classe para perto e
amostras do mesmo atributo sensível (mas de classes diferentes) para
longe, para evitar que a representação codifique o atributo demográfico.

Loss:
    L_FSCL+(z, y, a) = L_SupCon(z, y) + λ * L_debias(z, a, y)

O termo debias penaliza pares com mesmo atributo sensível a e classes
diferentes que ficaram próximos no espaço de embedding.

Uso principal como baseline de comparação — o backbone e a head são
compartilhados com as configs A/B/C para isolar o efeito da loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSCLPlusLoss(nn.Module):
    """FSCL+ loss (SupCon + termo de debias).

    Parâmetros
    ----------
    temperature:
        Temperatura da softmax contrastiva (default 0.07, conforme SupCon).
    lambda_debias:
        Peso do termo de debias. Park et al. usa 1.0.
    """

    def __init__(self, temperature: float = 0.07, lambda_debias: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_debias = float(lambda_debias)

    def forward(
        self,
        z: torch.Tensor,             # (N, D) — embeddings L2-normalizados
        labels: torch.Tensor,        # (N,) — classe (raça)
        sensitive: torch.Tensor,     # (N,) — atributo sensível (gênero, etc.)
    ) -> torch.Tensor:
        n = z.size(0)
        if n < 2:
            return z.new_zeros(())

        # matriz de similaridade
        sim = z @ z.t() / self.temperature
        # mask que zera a diagonal
        mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_self, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
        # Substitui -inf da diagonal por 0 (para não gerar NaN em máscaras adiante)
        log_prob = torch.where(mask_self, torch.zeros_like(log_prob), log_prob)

        # SupCon: pares positivos = mesmo label (excluindo self)
        same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
        pos_count = same_label.sum(-1).clamp(min=1)
        pos_log = torch.where(same_label, log_prob, torch.zeros_like(log_prob))
        supcon = -pos_log.sum(-1) / pos_count

        # Debias: penaliza atração entre same_sensitive & different_label
        same_sens = sensitive.unsqueeze(0) == sensitive.unsqueeze(1)
        diff_label = ~same_label & ~mask_self
        neg_bias = same_sens & diff_label
        neg_count = neg_bias.sum(-1).clamp(min=1)
        neg_log = torch.where(neg_bias, log_prob, torch.zeros_like(log_prob))
        debias = neg_log.sum(-1) / neg_count

        loss = supcon.mean() + self.lambda_debias * debias.mean()
        return loss
