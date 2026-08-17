"""Unit tests para src/face_bias/fairness/pareto.py — Etapa 4."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO Etapa 4 (Abr/2027)")
def test_pareto_front_simple_case():
    """3 pontos: (1,3), (2,2), (3,1) — todos na fronteira."""
    pass


@pytest.mark.skip(reason="TODO Etapa 4")
def test_pareto_front_dominated_point_excluded():
    """(5,5) dominado por (3,3) deve ser excluído da fronteira."""
    pass


@pytest.mark.skip(reason="TODO Etapa 4")
def test_pareto_plot_generates_file(tmp_path):
    pass
