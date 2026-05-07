"""Tests for the undersampling helper used to reproduce MBA Cap. 4."""

import pandas as pd
import pytest

from face_bias.data.dataset import _undersample_to_minority


@pytest.mark.unit
def test_undersample_makes_each_class_equal_to_minority() -> None:
    df = pd.DataFrame(
        {
            "file": [f"img_{i}.jpg" for i in range(50)],
            "race": ["A"] * 30 + ["B"] * 15 + ["C"] * 5,
        }
    )
    out = _undersample_to_minority(df, label_column="race", random_state=42)
    counts = out["race"].value_counts().to_dict()
    assert counts == {"A": 5, "B": 5, "C": 5}
    assert len(out) == 15


@pytest.mark.unit
def test_undersample_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "file": [f"img_{i}.jpg" for i in range(40)],
            "race": ["A"] * 20 + ["B"] * 10 + ["C"] * 10,
        }
    )
    a = _undersample_to_minority(df, label_column="race", random_state=7)
    b = _undersample_to_minority(df, label_column="race", random_state=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_undersample_preserves_columns() -> None:
    df = pd.DataFrame(
        {
            "file": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "race": ["X", "Y", "X", "Y"],
            "extra": [1, 2, 3, 4],
        }
    )
    out = _undersample_to_minority(df, label_column="race", random_state=0)
    assert list(out.columns) == ["file", "race", "extra"]
