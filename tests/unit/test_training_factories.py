"""Tests for the optimizer / scheduler factories and callbacks."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from face_bias.training.callbacks import EarlyStopping, ModelCheckpoint
from face_bias.training.optimizers import build_optimizer, parameter_group_count
from face_bias.training.schedulers import build_scheduler, is_step_per_batch


@pytest.fixture
def model():
    return nn.Linear(10, 7)


def _config(optimizer="adamw", scheduler="onecyclelr", lr=1e-3, epochs=2):
    return {
        "training": {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "learning_rate": lr,
            "num_epochs": epochs,
        }
    }


# ----- optimizer factory -----


@pytest.mark.unit
def test_build_optimizer_sgd(model) -> None:
    opt = build_optimizer(model.parameters(), _config(optimizer="sgd"))
    assert isinstance(opt, SGD)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(5e-4)


@pytest.mark.unit
def test_build_optimizer_adamw(model) -> None:
    opt = build_optimizer(model.parameters(), _config(optimizer="adamw"))
    assert isinstance(opt, AdamW)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(5e-4)


@pytest.mark.unit
def test_build_optimizer_unknown_raises(model) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model.parameters(), _config(optimizer="rmsprop"))


@pytest.mark.unit
def test_parameter_group_count(model) -> None:
    opt = build_optimizer(model.parameters(), _config())
    assert parameter_group_count(opt) == 1


# ----- scheduler factory -----


@pytest.mark.unit
def test_build_scheduler_onecyclelr(model) -> None:
    opt = build_optimizer(model.parameters(), _config())
    sched = build_scheduler(opt, _config(scheduler="onecyclelr"), steps_per_epoch=4)
    assert isinstance(sched, OneCycleLR)
    assert is_step_per_batch(sched)


@pytest.mark.unit
def test_build_scheduler_cosine(model) -> None:
    opt = build_optimizer(model.parameters(), _config())
    sched = build_scheduler(
        opt, _config(scheduler="cosineannealingwarmrestarts"), steps_per_epoch=4
    )
    assert isinstance(sched, CosineAnnealingWarmRestarts)
    assert not is_step_per_batch(sched)


@pytest.mark.unit
def test_build_scheduler_unknown_raises(model) -> None:
    opt = build_optimizer(model.parameters(), _config())
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(opt, _config(scheduler="exponential"), steps_per_epoch=4)


# ----- EarlyStopping -----


@pytest.mark.unit
def test_early_stopping_min_mode_triggers() -> None:
    es = EarlyStopping(patience=2, mode="min")
    assert es.step(1.0) is False  # baseline
    assert es.step(0.9) is False  # improved
    assert es.step(0.95) is False  # no improvement, count=1
    assert es.step(0.96) is True  # no improvement, count=2 -> stop
    assert es.should_stop is True


@pytest.mark.unit
def test_early_stopping_max_mode() -> None:
    es = EarlyStopping(patience=1, mode="max")
    es.step(0.5)  # baseline
    assert es.step(0.4) is True


@pytest.mark.unit
def test_early_stopping_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(mode="hopeful")


# ----- ModelCheckpoint -----


@pytest.mark.unit
def test_model_checkpoint_saves_on_improvement(tmp_path, model) -> None:
    cp = ModelCheckpoint(dirpath=tmp_path, mode="min")
    saved = cp.step(0.5, model)
    assert saved is True
    assert cp.path.exists()


@pytest.mark.unit
def test_model_checkpoint_does_not_save_when_worse(tmp_path, model) -> None:
    cp = ModelCheckpoint(dirpath=tmp_path, mode="min")
    cp.step(0.5, model)
    saved = cp.step(0.7, model)
    assert saved is False


@pytest.mark.unit
def test_model_checkpoint_payload_loadable(tmp_path, model) -> None:
    cp = ModelCheckpoint(dirpath=tmp_path, mode="min")
    cp.step(0.42, model, extra={"epoch": 3})
    payload = torch.load(cp.path, weights_only=False)
    assert payload["metric"] == pytest.approx(0.42)
    assert payload["epoch"] == 3
    assert "model_state_dict" in payload
