"""Unit tests for the t-SNE pipeline."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from face_bias.interpretability.tsne import compute_embeddings, plot_tsne, run_tsne


class _ToyModel(nn.Module):
    def __init__(self, in_features=4, embed_dim=8):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)

    def extract_features(self, x):
        return torch.relu(self.proj(x))

    def forward(self, x):
        return self.proj(x)


def _make_loader(n=40, in_features=4, n_classes=3) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, in_features)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


@pytest.mark.unit
def test_compute_embeddings_shapes() -> None:
    model = _ToyModel(in_features=4, embed_dim=8)
    loader = _make_loader(n=40)
    feats, labels = compute_embeddings(model, loader, torch.device("cpu"))
    assert feats.shape == (40, 8)
    assert labels.shape == (40,)


@pytest.mark.unit
def test_compute_embeddings_respects_max_samples() -> None:
    model = _ToyModel()
    loader = _make_loader(n=40)
    feats, labels = compute_embeddings(model, loader, torch.device("cpu"), max_samples=12)
    assert feats.shape[0] == 12
    assert labels.shape[0] == 12


@pytest.mark.unit
def test_run_tsne_returns_2d() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((30, 8)).astype(np.float32)
    coords = run_tsne(features, perplexity=10.0, n_iter=300)
    assert coords.shape == (30, 2)


@pytest.mark.unit
def test_plot_tsne_writes_png(tmp_path) -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((30, 2))
    labels = np.array([i % 3 for i in range(30)])
    out = tmp_path / "tsne.png"
    plot_tsne(coords, labels, ["a", "b", "c"], out)
    assert out.exists()
    assert out.with_suffix(".pdf").exists()
