"""Unit tests para src/face_bias/conditioning/film.py — Etapa 3.

Cobertura prevista:
    - shape preservation após FiLM (in == out)
    - init identidade (γ ≈ 1, β ≈ 0) faz FiLM se comportar como passthrough
    - gradient flow através de γ e β
    - Config B (linear) vs Config C (gated) — comportamento diferenciado
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO Etapa 3 (Jan-Mar/2027) — implementar src/face_bias/conditioning/film.py")
def test_film_shape_preservation():
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_film_identity_init():
    """FiLM inicializada com γ=1, β=0 deve retornar F_i inalterado."""
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_film_gradient_flow():
    pass


@pytest.mark.skip(reason="TODO Etapa 3")
def test_gated_film_differs_from_linear():
    pass
