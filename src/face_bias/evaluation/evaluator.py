"""Evaluator — runs a trained model over a DataLoader and emits a full report.

Output artefacts (saved under ``output_dir``):
- ``metrics.json``       aggregate accuracy / F1 / log-loss
- ``per_class.csv``      precision/recall/F1/support per class
- ``fairness_audit.json``  IR, max-min, std, CV, Gini for precision, recall, F1
- ``confusion_matrix.csv``  pandas DataFrame
- ``classification_report.txt``  sklearn-style text report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from face_bias.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_dataframe,
    fairness_audit,
    per_class_report,
    text_classification_report,
)


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred, y_proba)."""
    model.eval()
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits = (
            model(images, labels=labels_dev) if hasattr(model, "head_type") else model(images)
        )
        proba = torch.softmax(logits, dim=1)
        all_proba.append(proba.cpu().numpy())
        all_pred.append(proba.argmax(dim=1).cpu().numpy())
        all_target.append(labels.numpy())

    return (
        np.concatenate(all_target),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    class_names: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_proba = predict(model, dataloader, device)

    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)
    audit = fairness_audit(y_true, y_pred, class_names)
    report = per_class_report(y_true, y_pred, class_names)
    cm = confusion_matrix_dataframe(y_true, y_pred, class_names)
    report_text = text_classification_report(y_true, y_pred, class_names)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "fairness_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report.to_csv(output_dir / "per_class.csv", index=False)
    cm.to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    logging.info(
        "evaluate accuracy={accuracy:.4f} f1_macro={f1_macro:.4f} "
        "f1_IR={ir:.3f} f1_max_min={gap:.3f}".format(
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            ir=audit["f1"]["inequity_rate"],
            gap=audit["f1"]["max_min_disparity"],
        )
    )

    return {
        "metrics": metrics,
        "fairness": audit,
        "per_class": report.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }
