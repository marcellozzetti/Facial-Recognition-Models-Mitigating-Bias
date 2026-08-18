"""Unit tests para src/face_bias/mst/trainer.py — Etapa 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_bias.mst import stratified_split
from face_bias.mst.trainer import _f1_macro_per_class


@pytest.mark.unit
def test_stratified_split_preserves_all_classes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "file": [f"img_{i}.jpg" for i in range(200)],
        "mst_label": rng.integers(1, 11, size=200),
    })
    train_df, val_df = stratified_split(df, val_frac=0.2, seed=42)
    assert len(train_df) + len(val_df) == 200
    assert set(train_df["mst_label"]) == set(df["mst_label"])
    for label in df["mst_label"].unique():
        n_val = int((val_df["mst_label"] == label).sum())
        assert n_val >= 1  # cada classe tem pelo menos 1 em val


@pytest.mark.unit
def test_stratified_split_singleton_class_goes_to_train():
    df = pd.DataFrame({
        "file": ["x.jpg", "a.jpg", "b.jpg"],
        "mst_label": [1, 5, 5],
    })
    train_df, val_df = stratified_split(df, val_frac=0.2, seed=0)
    # classe 1 tem só 1 amostra: vai toda para train
    assert 1 in train_df["mst_label"].values
    assert 1 not in val_df["mst_label"].values


@pytest.mark.unit
def test_stratified_split_is_deterministic():
    df = pd.DataFrame({
        "file": [f"img_{i}.jpg" for i in range(100)],
        "mst_label": np.arange(100) % 10 + 1,
    })
    t1, v1 = stratified_split(df, val_frac=0.2, seed=42)
    t2, v2 = stratified_split(df, val_frac=0.2, seed=42)
    assert t1["file"].tolist() == t2["file"].tolist()
    assert v1["file"].tolist() == v2["file"].tolist()


@pytest.mark.unit
def test_f1_macro_perfect_prediction():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    assert _f1_macro_per_class(y_true, y_pred, n_classes=3) == pytest.approx(1.0)


@pytest.mark.unit
def test_f1_macro_missing_class_is_ignored():
    # classe 2 nunca aparece; F1 macro considera só classes com suporte
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    val = _f1_macro_per_class(y_true, y_pred, n_classes=10)
    assert val == pytest.approx(1.0)


@pytest.mark.unit
def test_f1_macro_all_wrong_is_zero():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])
    val = _f1_macro_per_class(y_true, y_pred, n_classes=3)
    assert val == pytest.approx(0.0)
