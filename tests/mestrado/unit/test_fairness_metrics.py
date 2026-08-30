"""Unit tests para src/face_bias/fairness/{disparity_ratio,worst_class_f1,
equal_opportunity,equalized_odds}.py — Etapa 4."""

from __future__ import annotations

import numpy as np
import pytest

from face_bias.fairness import (
    disparity_ratio,
    equal_opportunity_gap,
    equal_opportunity_per_class,
    equalized_odds_gap,
    equalized_odds_per_class,
    inequity_rate,
    per_class_f1,
    report_dr,
    worst_class_f1,
    worst_class_f1_bootstrap,
)


# ---------------------------------------------------------------------------
# per_class_f1 / disparity_ratio / worst_class_f1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_class_f1_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    scores = per_class_f1(y, y, n_classes=3)
    assert np.allclose(scores, [1.0, 1.0, 1.0])


@pytest.mark.unit
def test_per_class_f1_missing_class_returns_nan():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # classe 2 nunca aparece
    scores = per_class_f1(y_true, y_pred, n_classes=3)
    assert np.isnan(scores[2])


@pytest.mark.unit
def test_disparity_ratio_perfect_agreement_is_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    dr = disparity_ratio(y, y, n_classes=3)
    assert dr == pytest.approx(1.0)


@pytest.mark.unit
def test_disparity_ratio_severe_imbalance():
    # classe 1 é reconhecida muito pior que a classe 0 → DR longe de 1
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    dr = disparity_ratio(y_true, y_pred, n_classes=2)
    assert 0 < dr < 0.7  # F1_0 ~ 0.73, F1_1 ~ 0.4 → DR ~ 0.55


@pytest.mark.unit
def test_inequity_rate_is_complement_of_dr():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    dr = disparity_ratio(y_true, y_pred, n_classes=3)
    ir = inequity_rate(y_true, y_pred, n_classes=3)
    assert ir == pytest.approx(1 - dr, abs=1e-9)


@pytest.mark.unit
def test_worst_class_f1_returns_min():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 0, 0])
    wf1 = worst_class_f1(y_true, y_pred, n_classes=3)
    # classe 2 tem TP=1, FN=2, FP=0 → recall=1/3, precision=1 → F1=0.5
    assert 0 < wf1 <= 0.6


@pytest.mark.unit
def test_worst_class_f1_bootstrap_returns_ci():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = y_true.copy()
    # perturba 15% para F1 < 1
    idx = rng.choice(200, size=30, replace=False)
    y_pred[idx] = rng.integers(0, 3, size=30)
    point, lo, hi = worst_class_f1_bootstrap(y_true, y_pred, n_classes=3, n_boot=200)
    assert lo <= point <= hi
    assert 0 < point < 1


@pytest.mark.unit
def test_report_dr_flags_worst_and_best():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])  # classe 2 sempre errada
    df = report_dr(y_true, y_pred, class_names=["W", "B", "L"])
    row_l = df[df["class"] == "L"].iloc[0]
    assert row_l["is_worst"]
    assert not row_l["is_best"]


# ---------------------------------------------------------------------------
# equal_opportunity / equalized_odds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_equal_opportunity_gap_zero_when_perfect():
    # 4 amostras classe 1, distribuídas em 2 grupos, TPR igual
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0])
    group = np.array(["M", "F", "M", "F", "M", "F"])
    gap = equal_opportunity_gap(y_true, y_pred, group, target_class=1)
    assert gap == pytest.approx(0.0)


@pytest.mark.unit
def test_equal_opportunity_gap_detects_bias():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0])  # só grupo M acerta classe 1
    group = np.array(["M", "M", "F", "F", "M", "F"])
    gap = equal_opportunity_gap(y_true, y_pred, group, target_class=1)
    assert gap == pytest.approx(1.0)  # M=1.0, F=0.0


@pytest.mark.unit
def test_equal_opportunity_per_class_has_gap_column():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    group = np.array(["M", "F", "M", "F", "M", "F"])
    df = equal_opportunity_per_class(y_true, y_pred, group, n_classes=3)
    assert "eo_gap" in df.columns
    assert len(df) == 3
    assert np.allclose(df["eo_gap"], 0.0)


@pytest.mark.unit
def test_equalized_odds_stricter_than_eo():
    """EqOdds vê o gap de TPR e FPR simultaneamente — deve ser >= EO gap."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    # TPR: M=1, F=1 (gap 0); FPR: M=1, F=0 (gap 1)
    group = np.array(["M", "M", "M", "M", "F", "F", "F", "F"])
    eo = equal_opportunity_gap(y_true, y_pred, group, target_class=1)
    eq = equalized_odds_gap(y_true, y_pred, group, target_class=1)
    assert eq >= eo
    assert eq == pytest.approx(1.0)


@pytest.mark.unit
def test_equalized_odds_per_class_columns():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    group = np.array(["A", "B", "A", "B"])
    df = equalized_odds_per_class(y_true, y_pred, group, n_classes=2)
    assert "eqodds_gap" in df.columns
    for prefix in ("tpr_", "fpr_"):
        assert any(c.startswith(prefix) for c in df.columns)
