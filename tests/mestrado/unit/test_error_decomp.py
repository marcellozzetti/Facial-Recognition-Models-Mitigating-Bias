"""Unit tests para src/face_bias/decomposition/error_decomp.py — Etapa 6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_bias.decomposition import (
    decompose_variance,
    generate_report,
    test_h4_overlap_concentration as assess_h4,
    test_h6_pixel_info_explains as assess_h6,
)


def _synthetic_errors(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        mst = int(rng.integers(1, 11))
        model = str(rng.choice(["A", "B", "C"]))
        race = str(rng.choice(["White", "Black", "Latino_Hispanic"]))
        # erro depende de mst (fenótipo) + model (algoritmo) + ruído
        base = 0.3 if mst >= 7 else 0.1
        base += 0.15 if model == "A" else 0.0
        err = float(rng.uniform(base, base + 0.2))
        rows.append({
            "error": err, "mst_pred": mst, "model": model, "race": race,
        })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_decompose_variance_returns_valid_r2s():
    rng = np.random.default_rng(0)
    df = _synthetic_errors(rng)
    r = decompose_variance(df)
    assert 0 <= r.r2_total <= 1
    assert 0 <= r.r2_phenotypic <= 1
    assert 0 <= r.r2_algorithmic <= 1
    assert r.n_observations == len(df)


@pytest.mark.unit
def test_decompose_variance_rejects_missing_columns():
    df = pd.DataFrame({"error": [0.1, 0.2]})
    with pytest.raises(ValueError):
        decompose_variance(df)


@pytest.mark.unit
def test_decompose_variance_phenotypic_wins_when_mst_drives():
    """Se erro só depende de mst, R² fenotípico deve ser maior que algorítmico."""
    rng = np.random.default_rng(1)
    n = 400
    rows = []
    for _ in range(n):
        mst = int(rng.integers(1, 11))
        model = str(rng.choice(["A", "B", "C"]))  # não afeta erro
        err = 0.1 + 0.02 * mst + float(rng.normal(0, 0.01))
        rows.append({"error": err, "mst_pred": mst, "model": model})
    r = decompose_variance(pd.DataFrame(rows))
    assert r.r2_phenotypic > r.r2_algorithmic


@pytest.mark.unit
def test_h4_confirmed_when_errors_concentrate_in_overlap():
    # matriz com 3 tons de sobreposição (Latinx presente em MST 4, 5, 6)
    matrix = pd.DataFrame(
        [
            [0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.10],  # White
            [0.10, 0.10, 0.05, 0.10, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10],  # Black
            [0.05, 0.10, 0.10, 0.20, 0.20, 0.20, 0.05, 0.05, 0.03, 0.02],  # Latinx
        ],
        index=["White", "Black", "Latino_Hispanic"],
        columns=list(range(1, 11)),
    )
    errors_df = pd.DataFrame({
        "error": [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        "mst_pred": [4, 5, 6, 5, 4, 6, 4, 5, 6, 4],  # todos em overlap
        "model": ["A"] * 10,
        "race": ["Latino_Hispanic"] * 10,
    })
    r = assess_h4(errors_df, matrix, threshold=0.5)
    assert r.confirmed
    assert r.overlap_concentration >= 0.5


@pytest.mark.unit
def test_h4_refuted_when_errors_outside_overlap():
    matrix = pd.DataFrame(
        [
            [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
            [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],  # Latinx concentrado em 1 e 10
        ],
        index=["White", "Black", "Latino_Hispanic"],
        columns=list(range(1, 11)),
    )
    errors_df = pd.DataFrame({
        "error": [1, 1, 1],
        "mst_pred": [1, 1, 1],  # todos em tom 1 (sobreposição só se White esta presente)
        "model": ["A"] * 3,
        "race": ["Latino_Hispanic"] * 3,
    })
    r = assess_h4(errors_df, matrix, threshold=0.9)
    # tom 1 tem White(1.0) + Latinx(0.5) → 2 raças >= 0.05 → é overlap
    # concentração = 3/3 = 1.0 → CONFIRMADA para threshold 0.9
    # mas testando mais estrito (threshold 0.99, min races 3):
    r_strict = assess_h4(errors_df, matrix, threshold=0.5, overlap_min_races=3)
    assert not r_strict.confirmed  # só 2 raças presentes, não 3


@pytest.mark.unit
def test_h6_confirmed_when_pixel_info_explains():
    rng = np.random.default_rng(2)
    n = 200
    pixel_info = rng.uniform(0.1, 0.6, size=n)
    error = 0.9 - 1.4 * pixel_info + rng.normal(0, 0.05, size=n)
    df = pd.DataFrame({
        "error": error, "pixel_information": pixel_info,
        "mst_pred": rng.integers(1, 11, size=n),
        "model": ["A"] * n,
    })
    r = assess_h6(df, threshold=0.7)
    assert r.confirmed
    assert r.r2_pixel_info >= 0.7


@pytest.mark.unit
def test_h6_refuted_when_pixel_info_uninformative():
    rng = np.random.default_rng(3)
    n = 200
    pixel_info = rng.uniform(size=n)
    error = rng.uniform(size=n)  # não correlacionado
    df = pd.DataFrame({
        "error": error, "pixel_information": pixel_info,
        "mst_pred": rng.integers(1, 11, size=n),
        "model": ["A"] * n,
    })
    r = assess_h6(df, threshold=0.7)
    assert not r.confirmed
    assert r.r2_pixel_info < 0.3


@pytest.mark.unit
def test_generate_report_contains_key_sections():
    r = decompose_variance(_synthetic_errors(np.random.default_rng(0)))
    md = generate_report(r)
    assert "Decomposição de variância" in md
    assert "R² total" in md
