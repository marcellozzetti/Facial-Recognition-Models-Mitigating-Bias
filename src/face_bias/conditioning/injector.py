"""Integração FiLM ↔ ConvNeXt-T — pontos de inserção por estágio hierárquico.

Cap. 4 §4.5 e §4.6. ConvNeXt-T tem 4 estágios hierárquicos com canais
{96, 192, 384, 768}. Inserimos uma camada FiLM ao final de cada estágio.

Uso típico:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    wrapped = wrap_convnext_with_film(backbone, cond_dim=10, gated=False)
    logits = wrapped(images, z=mst_softmax)  # z opcional

Configurações do ablation (Cap. 4 §4.7):
    A (baseline) — backbone puro, ``wrap_convnext_with_film`` não é chamado
    B (proposta principal) — cond_dim=10, sinal MST direto
    C (CLIP-text) — cond_dim=512, sinal via ``CLIPPromptEnsembler``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from face_bias.conditioning.film import FiLMLayer

CONVNEXT_TINY_STAGE_CHANNELS = (96, 192, 384, 768)


class FiLMConditionedConvNeXt(nn.Module):
    """ConvNeXt-T com camadas FiLM inseridas após cada estágio hierárquico.

    O ``forward`` aceita ``z`` opcional:
      - ``z=None`` → comportamento retrocompatível (backbone puro).
      - ``z`` presente → modulação FiLM aplicada em todos os estágios.

    Parâmetros
    ----------
    backbone:
        Instância de ``torchvision.models.convnext_tiny``. Não é clonada;
        o wrapper toma posse.
    cond_dim:
        Dimensão do sinal condicionante (10 para MST, 512 para CLIP-text).
    hidden_dim:
        Dimensão oculta dos MLPs geradores de γ/β.
    gated:
        Se True, aplica porta sigmoide (variante de implementação da Config B).
    num_classes:
        Se > 0, adiciona head linear no final para classificação.
        Se None ou 0, retorna features globais (para uso com head externo).
    """

    def __init__(
        self,
        backbone: nn.Module,
        cond_dim: int = 10,
        hidden_dim: int = 128,
        gated: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.cond_dim = int(cond_dim)
        self.num_stages = len(CONVNEXT_TINY_STAGE_CHANNELS)

        self.film_layers = nn.ModuleList(
            [
                FiLMLayer(
                    cond_dim=cond_dim,
                    feature_channels=c,
                    hidden_dim=hidden_dim,
                    gated=gated,
                )
                for c in CONVNEXT_TINY_STAGE_CHANNELS
            ]
        )

        # torchvision ConvNeXt: features é Sequential com 8 elementos alternando
        # (downsample, stage, downsample, stage, downsample, stage, downsample, stage).
        # Índices dos stages (após cada downsample): 1, 3, 5, 7.
        self._stage_indices = (1, 3, 5, 7)

        # Head opcional
        if num_classes and num_classes > 0:
            emb_dim = CONVNEXT_TINY_STAGE_CHANNELS[-1]  # 768
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, num_classes),
            )
        else:
            self.head = None

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None and z.shape[1] != self.cond_dim:
            raise ValueError(
                f"z tem dim {z.shape[1]}, esperado {self.cond_dim}"
            )
        feats = x
        stage_ptr = 0
        for i, layer in enumerate(self.backbone.features):
            feats = layer(feats)
            if i in self._stage_indices:
                if z is not None:
                    feats = self.film_layers[stage_ptr](feats, z)
                stage_ptr += 1

        if self.head is not None:
            return self.head(feats)
        return feats

    def film_num_params(self) -> int:
        return sum(p.numel() for p in self.film_layers.parameters() if p.requires_grad)

    def backbone_num_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)


def wrap_convnext_with_film(
    backbone: nn.Module,
    cond_dim: int = 10,
    hidden_dim: int = 128,
    gated: bool = False,
    num_classes: Optional[int] = None,
) -> FiLMConditionedConvNeXt:
    """Envolve um ConvNeXt-T com camadas FiLM em cada estágio.

    Suporte inicial: apenas ``torchvision.models.convnext_tiny``.
    """
    if not hasattr(backbone, "features"):
        raise ValueError(
            "backbone deve expor .features (torchvision.models.convnext_tiny)."
        )
    return FiLMConditionedConvNeXt(
        backbone=backbone,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        gated=gated,
        num_classes=num_classes,
    )
