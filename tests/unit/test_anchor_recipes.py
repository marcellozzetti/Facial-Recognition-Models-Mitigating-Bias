"""Regression tests for the baseline-positioning anchor recipes
(FairFace and FineFACE). Locks: per-group LR splitting, Adam optimizer
alias, RandomCrop-padding train transform. See docs/baseline_positioning.md.
"""

import pytest
import torch
import torch.nn as nn

from face_bias.data.dataset import _build_transforms
from face_bias.models import LResNet50E_IR
from face_bias.training.optimizers import build_optimizer


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _cfg(**training_overrides):
    return {
        "training": {
            "optimizer": "sgd",
            "learning_rate": 0.002,
            **training_overrides,
        }
    }


# ---------- per-group LR (FineFACE-recipe anchor) ----------


@pytest.mark.unit
def test_lr_backbone_splits_param_groups_correctly() -> None:
    m = LResNet50E_IR(num_classes=7, head="linear", pretrained=False)
    cfg = _cfg(lr_backbone=2e-4)
    opt = build_optimizer(m, cfg)

    assert len(opt.param_groups) == 2, "expected 2 groups when lr_backbone set"
    # group with smaller LR = backbone, larger LR = head/other
    g_bb = min(opt.param_groups, key=lambda g: g["lr"])
    g_head = max(opt.param_groups, key=lambda g: g["lr"])
    assert g_bb["lr"] == 2e-4 and g_head["lr"] == 2e-3
    # backbone params must all belong to model.backbone
    bb_ids = {id(p) for p in m.backbone.parameters()}
    assert all(id(p) in bb_ids for p in g_bb["params"])
    assert any(id(p) not in bb_ids for p in g_head["params"]), (
        "head group must include non-backbone params (Linear head etc.)"
    )


@pytest.mark.unit
def test_no_lr_backbone_yields_single_group() -> None:
    m = LResNet50E_IR(num_classes=7, head="linear", pretrained=False)
    opt = build_optimizer(m, _cfg())  # no lr_backbone
    assert len(opt.param_groups) == 1


@pytest.mark.unit
def test_lr_backbone_with_iterable_params_raises_or_ignores() -> None:
    """If caller passes parameters() iterable (not a module), lr_backbone
    cannot be applied. The function should silently ignore (single group)
    rather than crash, to keep back-compat with old callers."""
    m = LResNet50E_IR(num_classes=7, head="linear", pretrained=False)
    opt = build_optimizer(m.parameters(), _cfg(lr_backbone=2e-4))
    assert len(opt.param_groups) == 1  # back-compat: ignored


# ---------- adam optimizer alias (FairFace-recipe anchor) ----------


@pytest.mark.unit
def test_adam_alias_uses_zero_weight_decay_by_default() -> None:
    m = LResNet50E_IR(num_classes=7, head="linear", pretrained=False)
    opt = build_optimizer(m, _cfg(optimizer="adam", learning_rate=1e-4))
    assert opt.defaults["weight_decay"] == 0.0


@pytest.mark.unit
def test_weight_decay_override_honored() -> None:
    m = LResNet50E_IR(num_classes=7, head="linear", pretrained=False)
    opt = build_optimizer(m, _cfg(weight_decay=0.0))
    assert opt.defaults["weight_decay"] == 0.0


# ---------- random crop padding (FineFACE-recipe aug) ----------


def _image_cfg(crop_pad=None):
    return {
        "image": {
            "image_size": [224, 224],
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        "training": (
            {"train_random_crop_padding": crop_pad}
            if crop_pad is not None
            else {}
        ),
    }


@pytest.mark.unit
def test_train_pipeline_has_random_crop_when_padding_set() -> None:
    t = _build_transforms(_image_cfg(crop_pad=8))
    train_op_names = [type(o).__name__ for o in t["train"].transforms]
    assert "RandomCrop" in train_op_names


@pytest.mark.unit
def test_train_pipeline_has_no_random_crop_when_padding_absent() -> None:
    t = _build_transforms(_image_cfg(crop_pad=None))
    train_op_names = [type(o).__name__ for o in t["train"].transforms]
    assert "RandomCrop" not in train_op_names


@pytest.mark.unit
def test_eval_pipeline_is_unaffected_by_crop_padding() -> None:
    """RandomCrop must NOT leak into the eval pipeline — eval stays
    deterministic Resize+ToTensor+Normalize."""
    t = _build_transforms(_image_cfg(crop_pad=8))
    eval_op_names = [type(o).__name__ for o in t["val"].transforms]
    assert "RandomCrop" not in eval_op_names
    assert eval_op_names == ["Resize", "ToTensor", "Normalize"]
