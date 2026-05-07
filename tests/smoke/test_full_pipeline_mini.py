"""End-to-end smoke test: face-bias-train then face-bias-evaluate on a mini dataset.

Avoids the 25M-param ResNet50 weight download by setting model.pretrained
= false. The test takes ~5-10s on CPU and exercises the full chain that
the dissertation's reproducibility relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from face_bias.cli import evaluate as eval_cli
from face_bias.cli import train as train_cli

CLASSES = ["A", "B", "C"]
N_PER_CLASS = 6
IMAGE_SIZE = (32, 32)


def _seed_dataset(root: Path) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for class_idx, class_name in enumerate(CLASSES):
        for i in range(N_PER_CLASS):
            filename = f"{class_name}_{i}.jpg"
            color = (50 + class_idx * 70, 100, 200 - class_idx * 70)
            Image.new("RGB", IMAGE_SIZE, color=color).save(images_dir / filename)
            rows.append({"file": filename, "race": class_name})

    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_config(root: Path, csv_path: Path, images_dir: Path) -> Path:
    config = {
        "model": {
            "name": "LResNet50E_IR",
            "pretrained": False,  # avoid the 100MB weight download
            "num_classes": len(CLASSES),
            "dropout": 0.0,
            "head": "linear",
            "arcface_s": 30.0,
            "arcface_m": 0.5,
            "arcface_easy_margin": False,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-3,
            "num_epochs": 1,
            "optimizer": "adamw",
            "scheduler": "onecyclelr",
            "loss_function": "cross_entropy",
            "test_size": 0.34,
            "num_workers": 0,
            "random_state": 42,
        },
        "image": {
            "image_size": [32, 32],
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        "preprocessing": {
            "processing_type": "detect_adjust_image",
            "num_workers": 1,
            "rotate_angle": 45,
            "num_rotations": 1,
            "max_faces_per_image": 1,
        },
        "data": {
            "dataset_file": str(csv_path),
            "dataset_image_input_path": str(images_dir),
            "dataset_image_output_path": str(images_dir),
        },
        "bucket": {
            "function": "download",
            "bucket_client": "s3",
            "bucket_path_file": str(root / "bucket"),
            "object_file_name": "x.zip",
            "bucket_name": "x",
        },
        "logging": {
            "log_dir": str(root / "logs"),
            "log_level": "INFO",
            "format": "%(asctime)s [%(run_id)s] %(message)s",
            "log_bucket_file": "bucket.log",
            "log_preprocessing_file": "preprocessing.log",
            "log_face_data_file": "face.log",
            "log_version_file": "version.log",
            "log_training_file": "training.log",
            "log_evaluation_file": "evaluation.log",
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.mark.smoke
def test_train_then_evaluate(tmp_path: Path) -> None:
    csv_path = _seed_dataset(tmp_path)
    config_path = _write_config(tmp_path, csv_path, tmp_path / "images")
    train_outputs = tmp_path / "train_out"
    eval_outputs = tmp_path / "eval_out"

    rc = train_cli.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(train_outputs),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0

    run_dirs = list(train_outputs.iterdir())
    assert len(run_dirs) == 1
    history = json.loads((run_dirs[0] / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    checkpoint = run_dirs[0] / "checkpoints" / "best.pt"
    assert checkpoint.exists()

    rc = eval_cli.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(eval_outputs),
            "--split",
            "test",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    eval_run_dirs = list(eval_outputs.iterdir())
    assert len(eval_run_dirs) == 1
    for name in (
        "metrics.json",
        "fairness_audit.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "classification_report.txt",
    ):
        assert (eval_run_dirs[0] / name).exists()
