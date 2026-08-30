"""Unit tests para src/face_bias/fairness/pareto.py — Etapa 4."""

from __future__ import annotations

import pytest

from face_bias.fairness import ModelResult, build_pareto_table, pareto_front


@pytest.mark.unit
def test_pareto_front_dominated_model_excluded():
    results = [
        ModelResult("dominated", macro_f1=0.60, inequity=0.30),
        ModelResult("winner", macro_f1=0.70, inequity=0.20),
    ]
    front = pareto_front(results)
    assert len(front) == 1
    assert front[0].name == "winner"


@pytest.mark.unit
def test_pareto_front_returns_all_when_incomparable():
    results = [
        ModelResult("fair_low", macro_f1=0.60, inequity=0.10),
        ModelResult("acc_high", macro_f1=0.85, inequity=0.35),
    ]
    front = pareto_front(results)
    assert len(front) == 2  # nenhum domina o outro


@pytest.mark.unit
def test_pareto_front_ordered_by_inequity():
    results = [
        ModelResult("m1", macro_f1=0.80, inequity=0.30),
        ModelResult("m2", macro_f1=0.75, inequity=0.15),
        ModelResult("m3", macro_f1=0.90, inequity=0.40),
    ]
    front = pareto_front(results)
    ineq = [r.inequity for r in front]
    assert ineq == sorted(ineq)


@pytest.mark.unit
def test_build_pareto_table_marks_pareto_column():
    results = [
        ModelResult("dominated", macro_f1=0.60, inequity=0.30),
        ModelResult("winner", macro_f1=0.70, inequity=0.20),
    ]
    table = build_pareto_table(results)
    assert table["pareto"].sum() == 1
    assert set(table.columns) == {"model", "macro_f1", "inequity", "pareto"}
