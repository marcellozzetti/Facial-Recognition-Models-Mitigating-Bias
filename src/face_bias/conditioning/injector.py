"""Integração FiLM ↔ ConvNeXt-T e CLIP ViT-B/16 (Etapa 2).

Cap. 4 §4.5 e §4.6. Duas variantes de backbone com FiLM:

  FiLMConditionedConvNeXt — única camada FiLM após Stage 4 do ConvNeXt-T
      (feature map 7×7×768). Overhead ≈14 000 params (<0,1% dos 28M do backbone).

  CLIPViTWithFiLM — FiLM aplicado ao token [CLS] após o último bloco do
      CLIP ViT-B/16, condicionado em z_mst (10-dim). Backbone CLIP pré-treinado
      (86M params) + FiLM + head Linear(512→7).

Configurações do ablation (Cap. 4 §4.7):
    A (baseline) — ConvNeXt-T puro, sem FiLM
    B (proposta principal) — ConvNeXt-T + FiLM(z_mst, 10-dim) no Stage 4
    C (CLIP+FiLM) — CLIP ViT-B/16 + FiLM(z_mst, 10-dim) no [CLS] token
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from face_bias.conditioning.film import FiLMLayer

CONVNEXT_TINY_STAGE_CHANNELS = (96, 192, 384, 768)
# torchvision ConvNeXt: backbone.features é Sequential de 8 elementos
# (stem+downsample, stage, downsample, stage, ...). Stage 4 está no índice 7.
_STAGE4_INDEX = 7


class FiLMConditionedConvNeXt(nn.Module):
    """ConvNeXt-T com única camada FiLM inserida após Stage 4 (7×7×768).

    O ``forward`` aceita ``z`` opcional:
      - ``z=None`` → comportamento retrocompatível (backbone puro).
      - ``z`` presente → modulação FiLM aplicada após Stage 4.

    Parâmetros
    ----------
    backbone:
        Instância de ``torchvision.models.convnext_tiny``. Não é clonada;
        o wrapper toma posse.
    cond_dim:
        Dimensão do sinal condicionante (10 para MST, 512 para CLIP-text).
    hidden_dim:
        Dimensão oculta do MLP gerador de γ/β.
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

        # Única camada FiLM após Stage 4 (768 canais, feature map 7×7).
        self.film_layer = FiLMLayer(
            cond_dim=cond_dim,
            feature_channels=CONVNEXT_TINY_STAGE_CHANNELS[-1],  # 768
            hidden_dim=hidden_dim,
            gated=gated,
        )

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
        for i, layer in enumerate(self.backbone.features):
            feats = layer(feats)
            if i == _STAGE4_INDEX and z is not None:
                feats = self.film_layer(feats, z)

        if self.head is not None:
            return self.head(feats)
        return feats

    def film_num_params(self) -> int:
        return sum(p.numel() for p in self.film_layer.parameters() if p.requires_grad)

    def backbone_num_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)


def wrap_convnext_with_film(
    backbone: nn.Module,
    cond_dim: int = 10,
    hidden_dim: int = 128,
    gated: bool = False,
    num_classes: Optional[int] = None,
) -> FiLMConditionedConvNeXt:
    """Envolve um ConvNeXt-T com camada FiLM única após Stage 4.

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


class CLIPViTWithFiLM(nn.Module):
    """Config C — CLIP ViT-B/16 com FiLM no token [CLS] após bloco 12.

    Cap. 4 §4.7 (Configuração C). Backbone: encoder visual do CLIP pré-treinado
    (openai/clip-vit-base-patch16, ~86M params). FiLM conditioning sobre z_mst
    (10-dim) aplicado ao token [CLS] logo antes da projeção final, mantendo
    inicialização identidade (γ≈1, β≈0) para não perturbar o backbone no início.

    Fluxo:
        x (B,3,224,224) → embeddings (B,197,768) → 12 blocos Transformer →
        FiLM([CLS], z_mst) → post_layernorm → visual_projection (768→512) →
        L2-norm → Linear(512→7)
    """

    CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
    VIT_HIDDEN = 768
    CLIP_PROJ_DIM = 512

    def __init__(
        self,
        cond_dim: int = 10,
        hidden_dim: int = 128,
        num_classes: int = 7,
    ):
        super().__init__()
        try:
            from transformers import CLIPModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Config C requer `pip install transformers`."
            ) from e

        clip = CLIPModel.from_pretrained(self.CLIP_MODEL_ID)
        vm = clip.vision_model
        self.embeddings = vm.embeddings
        self.pre_layrnorm = vm.pre_layrnorm
        self.encoder = vm.encoder
        self.post_layernorm = vm.post_layernorm
        self.visual_projection = clip.visual_projection

        self.film_layer = FiLMLayer(
            cond_dim=cond_dim,
            feature_channels=self.VIT_HIDDEN,
            hidden_dim=hidden_dim,
        )
        self.head = nn.Linear(self.CLIP_PROJ_DIM, num_classes)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        import torch.nn.functional as F
        hidden = self.embeddings(x)          # (B, 197, 768)
        hidden = self.pre_layrnorm(hidden)
        out = self.encoder(inputs_embeds=hidden)
        hidden = out.last_hidden_state        # (B, 197, 768)

        if z is not None:
            cls = hidden[:, 0, :]            # (B, 768)
            cls = self.film_layer(cls, z)    # FiLM on [CLS]
            hidden = torch.cat([cls.unsqueeze(1), hidden[:, 1:, :]], dim=1)

        cls_final = self.post_layernorm(hidden[:, 0, :])  # (B, 768)
        v = self.visual_projection(cls_final)              # (B, 512)
        v = F.normalize(v, dim=-1)
        return self.head(v)                                # (B, 7)

    def film_num_params(self) -> int:
        return sum(p.numel() for p in self.film_layer.parameters() if p.requires_grad)

    def backbone_num_params(self) -> int:
        film_ids = {id(p) for p in self.film_layer.parameters()}
        head_ids = {id(p) for p in self.head.parameters()}
        return sum(
            p.numel() for p in self.parameters()
            if id(p) not in film_ids and id(p) not in head_ids and p.requires_grad
        )
