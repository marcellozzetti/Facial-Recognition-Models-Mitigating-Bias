"""Integração FiLM ↔ ConvNeXt-T — pontos de inserção por estágio hierárquico.

Cap. 4, §4.5 (Escolha do backbone) e §4.6 (Racional da escolha).

ConvNeXt-T tem 4 estágios hierárquicos com canais C_i ∈ {96, 192, 384, 768}.
Inserimos uma camada FiLM após o bloco convolucional principal de cada
estágio, sem modificar depthwise convolutions ou inverted bottleneck blocks.

Interface pública prevista:
    def wrap_convnext_with_film(
        backbone: nn.Module,     # torchvision.models.convnext_tiny
        cond_dim: int = 10,
        gated: bool = False,     # Config B (False) ou Config C (True)
        clip_conditioning: bool = False,  # Config D usa embedding 512-dim
    ) -> nn.Module

    Retorna backbone modificado com forward assinatura:
        backbone.forward(x: Tensor, z: Tensor) -> Tensor

Config mapping:
    A (baseline) — wrap_convnext_with_film não é chamado (usar backbone puro)
    B (FiLM linear MST 10) — cond_dim=10, gated=False
    C (Gated FiLM MST 10)  — cond_dim=10, gated=True
    D (FiLM CLIP-text 512) — cond_dim=512, gated=False (+ clip_prompts wrapper externo)

TODO Etapa 3 (Jan-Mar/2027):
    [ ] Hook para localizar saída de cada stage do ConvNeXt-T
    [ ] Injetar FiLMLayer após cada bloco principal (não após downsample)
    [ ] Preservar assinatura original quando z=None (retrocompatibilidade)
    [ ] Testes: shape preservation, gradient flow through gamma/beta

Ver também:
    - src/face_bias/models/backbones.py — ConvNeXt-T já registrado (linha 14)
    - src/face_bias/conditioning/film.py — camada FiLM base
"""

from __future__ import annotations

# TODO: implementação
