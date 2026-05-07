"""Tests for face_bias.evaluation.metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from face_bias.evaluation.metrics import (
    classification_metrics,
    coefficient_of_variation,
    confusion_matrix_dataframe,
    fairness_audit,
    gini_coefficient,
    inequity_rate,
    max_min_disparity,
    per_class_report,
    text_classification_report,
)

CLASSES = [
    "Black",
    "East Asian",
    "Indian",
    "Latino Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]


@pytest.fixture
def perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    return y, y.copy()


@pytest.fixture
def biased_predictions():
    """Two classes perfect, one class always wrong."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])  # class 2 mapped to 0
    return y_true, y_pred


# -------- classification_metrics --------


@pytest.mark.unit
def test_classification_metrics_perfect(perfect_predictions) -> None:
    y_true, y_pred = perfect_predictions
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["f1_weighted"] == 1.0


@pytest.mark.unit
def test_classification_metrics_log_loss_with_proba() -> None:
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    pred = proba.argmax(axis=1)
    m = classification_metrics(y_true, pred, proba)
    assert "log_loss" in m
    assert m["log_loss"] > 0


# -------- per_class_report --------


@pytest.mark.unit
def test_per_class_report_columns() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    report = per_class_report(y, y, ["a", "b", "c"])
    assert list(report.columns) == ["class", "precision", "recall", "f1", "support"]
    assert (report["f1"] == 1.0).all()


@pytest.mark.unit
def test_per_class_report_biased(biased_predictions) -> None:
    y_true, y_pred = biased_predictions
    report = per_class_report(y_true, y_pred, ["a", "b", "c"])
    # Class 2 was always misclassified — recall=0 and f1=0.
    assert report.loc[report["class"] == "c", "recall"].iloc[0] == 0.0


# -------- fairness primitives --------


@pytest.mark.unit
def test_inequity_rate_perfectly_fair() -> None:
    assert inequity_rate([0.9, 0.9, 0.9]) == 1.0


@pytest.mark.unit
def test_inequity_rate_known_value() -> None:
    assert math.isclose(inequity_rate([0.5, 0.9]), 0.9 / 0.5, rel_tol=1e-9)


@pytest.mark.unit
def test_inequity_rate_zero_returns_inf() -> None:
    assert inequity_rate([0.0, 0.5]) == float("inf")


@pytest.mark.unit
def test_max_min_disparity() -> None:
    assert max_min_disparity([0.5, 0.9, 0.7]) == pytest.approx(0.4)


@pytest.mark.unit
def test_coefficient_of_variation_zero_when_uniform() -> None:
    assert coefficient_of_variation([0.7, 0.7, 0.7]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_gini_zero_when_uniform() -> None:
    assert gini_coefficient([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_gini_nonzero_when_unequal() -> None:
    assert gini_coefficient([0.0, 0.0, 0.0, 1.0]) > 0


# -------- fairness_audit --------


@pytest.mark.unit
def test_fairness_audit_structure(biased_predictions) -> None:
    y_true, y_pred = biased_predictions
    audit = fairness_audit(y_true, y_pred, ["a", "b", "c"])
    assert set(audit.keys()) == {"precision", "recall", "f1"}
    for column in audit.values():
        assert {"min", "max", "mean", "std", "inequity_rate", "max_min_disparity"} <= column.keys()


@pytest.mark.unit
def test_fairness_audit_perfect_predictions_have_ir_one(perfect_predictions) -> None:
    y_true, y_pred = perfect_predictions
    audit = fairness_audit(y_true, y_pred, [str(i) for i in range(7)])
    assert audit["f1"]["inequity_rate"] == 1.0
    assert audit["f1"]["max_min_disparity"] == 0.0


# -------- confusion matrix + report --------


@pytest.mark.unit
def test_confusion_matrix_dataframe_shape() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    df = confusion_matrix_dataframe(y, y, ["a", "b", "c"])
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.index) == ["a", "b", "c"]


@pytest.mark.unit
def test_text_classification_report_runs() -> None:
    y = np.array([0, 1, 2])
    text = text_classification_report(y, y, ["a", "b", "c"])
    assert "precision" in text
    assert "a" in text and "b" in text and "c" in text
