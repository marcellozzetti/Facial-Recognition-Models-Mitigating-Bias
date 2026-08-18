"""Unit tests para src/face_bias/mst/datasets.py — Etapa 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from face_bias.mst import (
    build_mst_dataset,
    class_balance,
    load_ccv2,
    load_mste,
    load_stw,
)


def _write_labels(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    labels = tmp_path / name / "labels.csv"
    labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(labels, index=False)
    return labels


@pytest.mark.unit
def test_load_mste_reads_valid_csv(tmp_path):
    _write_labels(tmp_path, "MSTE", [
        {"file": "images/a.jpg", "mst_label": 3},
        {"file": "images/b.jpg", "mst_label": 7},
    ])
    df = load_mste(tmp_path / "MSTE")
    assert len(df) == 2
    assert list(df["source"].unique()) == ["mste"]
    assert df["file"].iloc[0].endswith("a.jpg")


@pytest.mark.unit
def test_load_mste_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mste(tmp_path / "MSTE")


@pytest.mark.unit
def test_load_ccv2_reads_valid_csv(tmp_path):
    _write_labels(tmp_path, "CCv2", [
        {"file": "frames/x.jpg", "mst_label": 5},
    ])
    df = load_ccv2(tmp_path / "CCv2")
    assert len(df) == 1
    assert df["source"].iloc[0] == "ccv2"


@pytest.mark.unit
def test_load_stw_missing_raises_with_helpful_msg(tmp_path):
    with pytest.raises(FileNotFoundError, match="ainda não publicado"):
        load_stw(tmp_path / "STW")


@pytest.mark.unit
def test_invalid_labels_filtered_out(tmp_path):
    _write_labels(tmp_path, "MSTE", [
        {"file": "a.jpg", "mst_label": 3},
        {"file": "b.jpg", "mst_label": 0},   # fora de [1,10]
        {"file": "c.jpg", "mst_label": 11},  # fora de [1,10]
    ])
    df = load_mste(tmp_path / "MSTE")
    assert len(df) == 1
    assert df["mst_label"].iloc[0] == 3


@pytest.mark.unit
def test_build_mst_dataset_merges_and_dedupes(tmp_path):
    _write_labels(tmp_path, "MSTE", [
        {"file": "a.jpg", "mst_label": 2},
        {"file": "b.jpg", "mst_label": 8},
    ])
    _write_labels(tmp_path, "CCv2", [
        {"file": "c.jpg", "mst_label": 5},
    ])
    df = build_mst_dataset([
        {"source": "mste", "root": tmp_path / "MSTE"},
        {"source": "ccv2", "root": tmp_path / "CCv2"},
    ])
    assert len(df) == 3
    assert set(df["source"]) == {"mste", "ccv2"}


@pytest.mark.unit
def test_build_mst_dataset_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        build_mst_dataset([{"source": "unknown", "root": tmp_path}])


@pytest.mark.unit
def test_class_balance_shape_and_zero_fill():
    df = pd.DataFrame({"mst_label": [1, 1, 5, 10]})
    bal = class_balance(df)
    assert len(bal) == 10
    assert bal.loc[1] == 2
    assert bal.loc[5] == 1
    assert bal.loc[10] == 1
    assert bal.loc[3] == 0  # tom não presente
