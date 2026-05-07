"""Tests for face_bias.utils.reproducibility."""

import os
import random

import numpy as np
import pytest
import torch

from face_bias.utils.reproducibility import seed_everything, seed_from_config


@pytest.mark.unit
def test_seed_everything_returns_seed() -> None:
    assert seed_everything(123, deterministic=False) == 123


@pytest.mark.unit
def test_seed_everything_makes_python_random_deterministic() -> None:
    seed_everything(42, deterministic=False)
    a = [random.random() for _ in range(5)]
    seed_everything(42, deterministic=False)
    b = [random.random() for _ in range(5)]
    assert a == b


@pytest.mark.unit
def test_seed_everything_makes_numpy_deterministic() -> None:
    seed_everything(42, deterministic=False)
    a = np.random.rand(8)
    seed_everything(42, deterministic=False)
    b = np.random.rand(8)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_seed_everything_makes_torch_deterministic() -> None:
    seed_everything(42, deterministic=False)
    a = torch.randn(4, 4)
    seed_everything(42, deterministic=False)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)


@pytest.mark.unit
def test_seed_everything_sets_pythonhashseed() -> None:
    seed_everything(7, deterministic=False)
    assert os.environ["PYTHONHASHSEED"] == "7"


@pytest.mark.unit
def test_seed_from_config_returns_none_when_missing() -> None:
    assert seed_from_config({}) is None
    assert seed_from_config({"training": {}}) is None


@pytest.mark.unit
def test_seed_from_config_uses_training_random_state() -> None:
    assert seed_from_config({"training": {"random_state": 1234}}) == 1234
