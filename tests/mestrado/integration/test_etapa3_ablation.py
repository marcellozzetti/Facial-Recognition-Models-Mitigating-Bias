"""Integration test para o loop A/B/C/D de conditioning — Etapa 3.

Valida que as 4 configurações rodam end-to-end sobre um mini-batch
sintético, sem crashes, com shapes esperados.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO Etapa 3 (Jan-Mar/2027) — precisa FiLM + injector")
def test_config_A_baseline_forward_pass():
    """ConvNeXt-T sem conditioning — controle."""
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_config_B_film_linear_forward_pass():
    """ConvNeXt-T + FiLM linear com MST 10-dim."""
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_config_C_film_gated_forward_pass():
    """ConvNeXt-T + FiLM gated (sigmoid gate)."""
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_config_D_film_clip_forward_pass():
    """ConvNeXt-T + FiLM sobre CLIP-text 512-dim."""
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_ablation_configs_produce_different_gradients():
    """A, B, C, D devem gerar gradientes distintos (não são idênticos)."""
    pass
