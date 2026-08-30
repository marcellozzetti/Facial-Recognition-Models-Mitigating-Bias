"""Adversarial debiasing — Zhang et al. (2018).

Cap. 4 §Baselines. Segundo modelo (adversário) tenta prever o atributo
sensível a partir das features internas do classificador principal; o
principal é treinado para MAXIMIZAR a perda do adversário.

Fluxo:
    features = encoder(x)
    logits   = classifier(features)
    adv_pred = adversary(gradient_reverse(features))

    L_total = L_class - λ * L_adversary
    (o gradient reverse layer inverte o sinal do gradiente no backward)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    """Camada identidade no forward, inverte o gradiente no backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, lambda_)


class AdversarialHead(nn.Module):
    """Cabeça adversária simples: MLP prevê o atributo sensível."""

    def __init__(self, in_dim: int, n_sensitive: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_sensitive),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdversarialDebiasLoss(nn.Module):
    """Combina cross-entropy da tarefa principal com a perda adversária."""

    def __init__(self, lambda_adv: float = 1.0):
        super().__init__()
        self.lambda_adv = float(lambda_adv)

    def forward(
        self,
        logits_main: torch.Tensor,      # (N, C_main)
        y_main: torch.Tensor,           # (N,)
        logits_adv: torch.Tensor,       # (N, C_sensitive)
        y_sensitive: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        loss_main = F.cross_entropy(logits_main, y_main)
        loss_adv = F.cross_entropy(logits_adv, y_sensitive)
        # GRL já inverteu o gradiente; aqui somamos normalmente.
        return loss_main + self.lambda_adv * loss_adv
