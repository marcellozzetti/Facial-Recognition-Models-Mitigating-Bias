"""FineFACE — Cross-layer Mutual Attention Learning (Manzoor et al. 2024).

Cap. 4 §Baselines. Arquitetura Pareto-eficiente com módulo de atenção
mútua entre camadas de diferentes profundidades — camadas rasas
"olham" para camadas profundas e vice-versa, permitindo troca de
informação em múltiplas escalas.

Implementação simplificada como wrapper: acopla um bloco de atenção
entre duas features intermediárias do backbone, sem reimplementar
todo o pipeline original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayerMutualAttention(nn.Module):
    """Atenção mútua entre duas features (canais compatíveis via projeção)."""

    def __init__(self, dim_low: int, dim_high: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = int(n_heads)
        # projeção comum (usa a dimensão maior)
        self.hidden = max(dim_low, dim_high)
        self.proj_low = nn.Linear(dim_low, self.hidden)
        self.proj_high = nn.Linear(dim_high, self.hidden)
        self.attn_low = nn.MultiheadAttention(self.hidden, n_heads, batch_first=True)
        self.attn_high = nn.MultiheadAttention(self.hidden, n_heads, batch_first=True)
        self.out_low = nn.Linear(self.hidden, dim_low)
        self.out_high = nn.Linear(self.hidden, dim_high)
        self.norm_low = nn.LayerNorm(dim_low)
        self.norm_high = nn.LayerNorm(dim_high)

    @staticmethod
    def _flatten(feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        # (N, C, H, W) → (N, H*W, C)
        n, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2), (n, c, h, w)

    @staticmethod
    def _unflatten(seq: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        n, c, h, w = shape
        return seq.transpose(1, 2).view(n, c, h, w)

    def forward(
        self,
        feat_low: torch.Tensor,   # (N, C1, H1, W1)
        feat_high: torch.Tensor,  # (N, C2, H2, W2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_low, shape_low = self._flatten(feat_low)
        seq_high, shape_high = self._flatten(feat_high)

        p_low = self.proj_low(seq_low)
        p_high = self.proj_high(seq_high)

        low_new, _ = self.attn_low(p_low, p_high, p_high)  # low q, high k/v
        high_new, _ = self.attn_high(p_high, p_low, p_low)  # high q, low k/v

        seq_low2 = self.out_low(low_new) + seq_low
        seq_high2 = self.out_high(high_new) + seq_high

        feat_low2 = self._unflatten(seq_low2, shape_low)
        feat_high2 = self._unflatten(seq_high2, shape_high)

        # LayerNorm sobre canais
        feat_low2 = self.norm_low(feat_low2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_high2 = self.norm_high(feat_high2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return feat_low2, feat_high2
