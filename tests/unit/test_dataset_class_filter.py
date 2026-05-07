"""Tests for the class-filter helper used to reproduce MBA Exp. 7/8."""

import pandas as pd
import pytest

from face_bias.data.dataset import _filter_by_class


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": [f"{i}.jpg" for i in range(10)],
            "race": ["Black", "White", "Indian", "Black", "White", "East Asian"] + ["Black"] * 4,
        }
    )


@pytest.mark.unit
def test_filter_keeps_only_allowed(sample_df) -> None:
    out = _filter_by_class(sample_df, "race", ["Black", "White"])
    assert set(out["race"].unique()) == {"Black", "White"}
    # 6 Black + 2 White in the fixture
    assert len(out) == 8


@pytest.mark.unit
def test_filter_none_returns_unchanged(sample_df) -> None:
    out = _filter_by_class(sample_df, "race", None)
    pd.testing.assert_frame_equal(out, sample_df)


@pytest.mark.unit
def test_filter_empty_list_returns_unchanged(sample_df) -> None:
    out = _filter_by_class(sample_df, "race", [])
    pd.testing.assert_frame_equal(out, sample_df)


@pytest.mark.unit
def test_filter_unknown_class_yields_empty(sample_df) -> None:
    out = _filter_by_class(sample_df, "race", ["Klingon"])
    assert len(out) == 0
