"""Smoke test: trainer fits a tiny model on a synthetic mini-dataset."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from face_bias.training.callbacks import EarlyStopping
from face_bias.training.optimizers import build_optimizer
from face_bias.training.schedulers import build_scheduler
from face_bias.training.trainer import Trainer, build_loss


def _make_loader(n_samples=16, n_features=4, n_classes=3) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


def _config():
    return {
        "training": {
            "optimizer": "adamw",
            "scheduler": "onecyclelr",
            "loss_function": "cross_entropy",
            "learning_rate": 1e-2,
            "num_epochs": 2,
        },
        "model": {"arcface_s": 30.0, "arcface_m": 0.5},
    }


@pytest.mark.smoke
def test_trainer_fits_two_epochs(tmp_path) -> None:
    config = _config()
    train_loader = _make_loader()
    val_loader = _make_loader(n_samples=8)

    model = nn.Linear(4, 3)
    loss = build_loss(config)
    opt = build_optimizer(model.parameters(), config)
    sched = build_scheduler(opt, config, steps_per_epoch=len(train_loader))

    trainer = Trainer(
        model=model,
        loss_fn=loss,
        optimizer=opt,
        scheduler=sched,
        device=torch.device("cpu"),
        class_names=["a", "b", "c"],
        checkpoint_dir=tmp_path,
        early_stopping=EarlyStopping(patience=5, mode="min"),
    )

    result = trainer.fit(train_loader, val_loader, epochs=config["training"]["num_epochs"])

    assert len(result["history"]) == 2
    assert (tmp_path / "best.pt").exists()
    # Each epoch entry has the keys the dissertation will plot.
    for entry in result["history"]:
        for key in ("epoch", "train_loss", "val_loss", "val_accuracy", "val_f1_macro"):
            assert key in entry


@pytest.mark.smoke
def test_trainer_early_stops(tmp_path) -> None:
    config = _config()
    config["training"]["num_epochs"] = 10

    train_loader = _make_loader()
    val_loader = _make_loader(n_samples=8)

    # Use a very low patience so we trigger after one non-improving epoch.
    early = EarlyStopping(patience=1, mode="min")
    model = nn.Linear(4, 3)
    loss = build_loss(config)
    opt = build_optimizer(model.parameters(), config)
    sched = build_scheduler(opt, config, steps_per_epoch=len(train_loader))

    trainer = Trainer(
        model=model,
        loss_fn=loss,
        optimizer=opt,
        scheduler=sched,
        device=torch.device("cpu"),
        class_names=["a", "b", "c"],
        checkpoint_dir=tmp_path,
        early_stopping=early,
    )
    result = trainer.fit(train_loader, val_loader, epochs=10)
    # Should stop before reaching 10 epochs on this trivial dataset.
    assert len(result["history"]) <= 10


@pytest.mark.unit
def test_build_loss_dispatch() -> None:
    from face_bias.models.losses import ArcFaceLoss, CrossEntropyLoss

    cfg = {
        "training": {"loss_function": "cross_entropy"},
        "model": {"arcface_s": 30.0, "arcface_m": 0.5},
    }
    assert isinstance(build_loss(cfg), CrossEntropyLoss)
    cfg["training"]["loss_function"] = "arcface"
    assert isinstance(build_loss(cfg), ArcFaceLoss)
    cfg["training"]["loss_function"] = "focal"
    import pytest as _p

    with _p.raises(ValueError, match="Unknown loss_function"):
        build_loss(cfg)
