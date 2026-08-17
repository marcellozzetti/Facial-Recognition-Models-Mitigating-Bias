"""Unit tests para src/face_bias/audit/*.py — Etapa 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_bias.audit import (
    FAIRFACE_RACES,
    MST_LABELS,
    build_matrix,
    coefficient_variation,
    entropy_per_race,
    join_predictions,
    spread_per_race,
    summarize,
    assess_hypothesis_h3,
)


def _make_audit_df(rng: np.random.Generator, n_per_race: int = 100) -> pd.DataFrame:
    """Cria audit_df sintético com todas as 7 raças e MST em 1..10."""
    rows = []
    for race in FAIRFACE_RACES:
        # Distribuição concentrada por raça, com pequeno ruído
        if race == "Latino_Hispanic":
            probs = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.03, 0.01, 0.01])
        elif race in ("White", "East Asian"):
            probs = np.array([0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        elif race == "Black":
            probs = np.array([0.002, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.30])
        else:
            probs = np.ones(10) / 10
        probs = probs / probs.sum()
        picks = rng.choice(MST_LABELS, size=n_per_race, p=probs)
        for m in picks:
            rows.append({"race": race, "mst_pred": int(m)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cross_matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_matrix_shape_and_normalization():
    rng = np.random.default_rng(0)
    df = _make_audit_df(rng)
    m = build_matrix(df)
    assert list(m.index) == FAIRFACE_RACES
    assert list(m.columns) == MST_LABELS
    np.testing.assert_allclose(m.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


@pytest.mark.unit
def test_build_matrix_handles_missing_race_or_mst():
    df = pd.DataFrame({"race": ["White", "Black"], "mst_pred": [2, 8]})
    m = build_matrix(df)
    # todas as 7 raças presentes, todas as 10 MST cols presentes
    assert m.shape == (7, 10)
    # linhas sem dados ficam com zeros (dividido por NaN -> 0)
    assert m.loc["Latino_Hispanic"].sum() == 0.0


@pytest.mark.unit
def test_spread_and_entropy_extremes():
    # matriz sintética: linha 0 = concentrada (só MST 5), linha 1 = uniforme
    idx = ["ConcentradaX", "UniformeY"]
    data = np.zeros((2, 10), dtype=np.float64)
    data[0, 4] = 1.0  # tudo em MST 5
    data[1, :] = 0.1  # uniforme
    m = pd.DataFrame(data, index=idx, columns=MST_LABELS)
    s = spread_per_race(m, threshold=0.05)
    assert s.loc["ConcentradaX"] == 1
    assert s.loc["UniformeY"] == 10
    h = entropy_per_race(m)
    assert h.loc["ConcentradaX"] == pytest.approx(0.0)
    assert h.loc["UniformeY"] == pytest.approx(np.log2(10), rel=1e-6)


@pytest.mark.unit
def test_coefficient_variation_positive():
    rng = np.random.default_rng(1)
    df = _make_audit_df(rng)
    m = build_matrix(df)
    cv = coefficient_variation(m)
    assert (cv >= 0).all()
    # linhas concentradas devem ter CV maior que a Latinx (mais espalhada)
    assert cv.loc["White"] > cv.loc["Latino_Hispanic"]


@pytest.mark.unit
def test_h3_confirmed_when_latinx_is_spread():
    rng = np.random.default_rng(2)
    df = _make_audit_df(rng, n_per_race=500)
    m = build_matrix(df)
    result = assess_hypothesis_h3(m, threshold=0.05, min_spread_required=5)
    assert result.latinx_spread >= 5
    assert result.confirmed is True
    d = result.as_dict()
    assert d["hypothesis"] == "H3"
    assert d["per_race_spread"]["Latino_Hispanic"] == result.latinx_spread


@pytest.mark.unit
def test_h3_refuted_when_latinx_concentrated():
    df = pd.DataFrame(
        {
            "race": ["Latino_Hispanic"] * 100 + ["White"] * 100,
            "mst_pred": [3] * 100 + [1] * 100,
        }
    )
    m = build_matrix(df)
    result = assess_hypothesis_h3(m, threshold=0.05, min_spread_required=5)
    assert result.latinx_spread == 1
    assert result.confirmed is False


@pytest.mark.unit
def test_summarize_columns():
    rng = np.random.default_rng(3)
    df = _make_audit_df(rng)
    m = build_matrix(df)
    r = summarize(m)
    assert set(r.columns) == {"spread", "entropy_bits", "cv"}
    assert list(r.index) == FAIRFACE_RACES


# ---------------------------------------------------------------------------
# fairface_mst
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_join_predictions_basename_fallback(tmp_path):
    pred = pd.DataFrame(
        {
            "path": [str(tmp_path / "img_A.jpg"), str(tmp_path / "img_B.jpg")],
            "mst_pred": [3, 7],
        }
    )
    lab = pd.DataFrame(
        {"file": ["val/img_A.jpg", "val/img_B.jpg"], "race": ["White", "Black"]}
    )
    merged = join_predictions(pred, lab, dataset_root=None)
    assert len(merged) == 2
    assert set(merged["race"].tolist()) == {"White", "Black"}


@pytest.mark.unit
def test_join_predictions_empty_raises(tmp_path):
    pred = pd.DataFrame({"path": [str(tmp_path / "img_A.jpg")], "mst_pred": [3]})
    lab = pd.DataFrame({"file": ["img_XX.jpg"], "race": ["White"]})
    with pytest.raises(RuntimeError):
        join_predictions(pred, lab, dataset_root=None)


@pytest.mark.unit
def test_load_fairface_labels_missing_columns(tmp_path):
    from face_bias.audit import load_fairface_labels
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fairface_labels(bad)
