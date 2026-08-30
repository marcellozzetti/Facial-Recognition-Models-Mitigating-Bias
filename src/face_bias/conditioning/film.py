"""Feature-wise Linear Modulation (FiLM) — mecanismo de condicionamento arquitetural.

Cap. 4 §4.6 (Formulação matemática) — proposta principal da pesquisa
(Contribuição 3, Cap. 3).

Referência canônica: Perez, Strub, de Vries, Dumoulin & Courville (2018),
AAAI Conference.

Formulação:
    FiLM(F | γ, β) = γ ⊙ F + β
    onde γ = f_γ(z), β = f_β(z), z ∈ R^cond_dim

A "linearidade" citada em outras partes do documento refere-se à
aplicação canal-a-canal da transformação afim — as MLPs geradoras de
γ e β podem ter camadas ocultas não-lineares. Variantes de
implementação (com/sem gating multiplicativo, profundidade da MLP)
ficam como escolhas empíricas dentro da Configuração B do ablation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPFilmGenerator(nn.Module):
    """MLP simples que projeta o sinal de contexto z em (γ, β).

    Init identidade: pesos zerados no gerador de γ e viés = 1 (γ ≈ 1
    no início do treino) e viés = 0 no gerador de β. Isso garante que
    FiLM começa como transformação identidade e o backbone pré-treinado
    não é perturbado no início do fine-tuning.
    """

    def __init__(
        self,
        cond_dim: int,
        feature_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.feature_channels = int(feature_channels)

        act_cls: type[nn.Module]
        if activation == "relu":
            act_cls = nn.ReLU
        elif activation == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"activation inválida: {activation!r}")

        self.trunk = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
        )
        self.gamma_head = nn.Linear(hidden_dim, self.feature_channels)
        self.beta_head = nn.Linear(hidden_dim, self.feature_channels)

        # Init identidade: γ ≈ 1, β ≈ 0 no início
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        return self.gamma_head(h), self.beta_head(h)


class FiLMLayer(nn.Module):
    """Camada FiLM aplicada sobre feature map (N, C, H, W) ou (N, C).

    Aceita variante ``gated=True``: aplica uma porta sigmoide sobre
    a saída modulada, ``out = σ(g(z)) ⊙ (γ ⊙ F + β)``. Usado como
    escolha empírica dentro da Configuração B — não é config separada.
    """

    def __init__(
        self,
        cond_dim: int,
        feature_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gated: bool = False,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.feature_channels = int(feature_channels)
        self.gated = bool(gated)

        self.generator = MLPFilmGenerator(
            cond_dim=cond_dim,
            feature_channels=feature_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if self.gated:
            self.gate_head = nn.Linear(hidden_dim, feature_channels)
            nn.init.zeros_(self.gate_head.weight)
            # bias 4 => sigmoid(4) ≈ 0.98 (porta ~aberta no início)
            nn.init.constant_(self.gate_head.bias, 4.0)

    def _reshape_params(self, params: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 4:
            return params.view(-1, self.feature_channels, 1, 1)
        if feature.dim() == 2:
            return params
        raise ValueError(f"esperado (N,C) ou (N,C,H,W); recebido {tuple(feature.shape)}")

    def forward(self, feature: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"canais esperados {self.feature_channels}, recebido {feature.shape[1]}"
            )
        gamma, beta = self.generator(z)
        gamma_r = self._reshape_params(gamma, feature)
        beta_r = self._reshape_params(beta, feature)
        modulated = gamma_r * feature + beta_r

        if self.gated:
            trunk_out = self.generator.trunk(z)
            gate = torch.sigmoid(self.gate_head(trunk_out))
            gate_r = self._reshape_params(gate, feature)
            modulated = gate_r * modulated

        return modulated

    @torch.no_grad()
    def is_identity_at_init(self, tolerance: float = 1e-5) -> bool:
        """Sanity: verifica que a camada começa como identidade."""
        z = torch.zeros(1, self.cond_dim)
        gamma, beta = self.generator(z)
        gate_ok = True
        if self.gated:
            trunk_out = self.generator.trunk(z)
            gate = torch.sigmoid(self.gate_head(trunk_out))
            gate_ok = bool(torch.all(gate > 0.95))
        return (
            torch.allclose(gamma, torch.ones_like(gamma), atol=tolerance)
            and torch.allclose(beta, torch.zeros_like(beta), atol=tolerance)
            and gate_ok
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_film_overhead(
    cond_dim: int,
    channels: list[int],
    hidden_dim: int = 128,
    gated: bool = False,
) -> dict:
    """Retorna número total de parâmetros das camadas FiLM inseridas em
    cada estágio (útil para relatar no texto o overhead ~1,3%).
    """
    total = 0
    per_stage = []
    for c in channels:
        layer = FiLMLayer(cond_dim, c, hidden_dim=hidden_dim, gated=gated)
        n = layer.num_params()
        total += n
        per_stage.append((c, n))
    return {"total_params": total, "per_stage": per_stage}
