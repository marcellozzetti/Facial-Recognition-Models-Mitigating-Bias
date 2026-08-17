"""Smoke test end-to-end do pipeline mestrado — todas as 6 Etapas.

Roda sobre mini-dataset (50-100 imagens) apenas para validar que o
grafo de dependências resolve e nenhuma etapa quebra na sequência
completa. Não valida qualidade científica (essa é feita nos
integration tests de cada etapa).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO Fase 0 tardia (Set-Out/2026) - após Etapas 1 e 3 iniciais")
def test_pipeline_end_to_end_mini():
    """
    Etapa 1: SkinToneNet infere sobre 50 imagens
    Etapa 2: matriz MST × raça é gerada
    Etapa 3: treina Config A (baseline, 1 epoch)
    Etapa 4: métricas triangulação são calculadas
    Etapa 5: transferência mock RFW funciona
    Etapa 6: decomposição roda sem crash
    """
    pass
