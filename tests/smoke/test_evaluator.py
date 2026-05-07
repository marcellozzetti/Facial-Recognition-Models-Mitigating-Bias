"""Smoke test: evaluator produces all output artefacts."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from face_bias.evaluation.evaluator import evaluate, predict


def _make_loader(n_samples=20, n_features=4, n_classes=3, seed=0) -> DataLoader:
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


@pytest.mark.smoke
def test_predict_shapes() -> None:
    loader = _make_loader()
    model = nn.Linear(4, 3)
    y_true, y_pred, y_proba = predict(model, loader, torch.device("cpu"))
    assert y_true.shape == (20,)
    assert y_pred.shape == (20,)
    assert y_proba.shape == (20, 3)
    # softmax rows sum to 1.
    assert pytest.approx(y_proba.sum(axis=1).mean(), abs=1e-5) == 1.0


@pytest.mark.smoke
def test_evaluate_writes_all_artifacts(tmp_path: Path) -> None:
    loader = _make_loader()
    model = nn.Linear(4, 3)
    out = evaluate(
        model,
        loader,
        torch.device("cpu"),
        class_names=["a", "b", "c"],
        output_dir=tmp_path,
    )

    expected = {
        "metrics.json",
        "fairness_audit.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "classification_report.txt",
    }
    assert {p.name for p in tmp_path.iterdir()} >= expected

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert "f1_macro" in metrics

    audit = json.loads((tmp_path / "fairness_audit.json").read_text(encoding="utf-8"))
    assert {"precision", "recall", "f1"} <= audit.keys()

    assert out["output_dir"] == str(tmp_path)
