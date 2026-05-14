"""Tests for the configurable MLP classification head (diretriz nº 2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from face_bias.models import LResNet50E_IR, MLPHead
from face_bias.models.mlp_head import _build_activation, _build_norm

NUM_CLASSES = 7
BATCH = 4
EMBED_DIM = 2048


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------- topology / construction ----------


@pytest.mark.unit
def test_mlp_head_single_hidden_layer_shape() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[512])
    out = head(torch.randn(BATCH, EMBED_DIM))
    assert out.shape == (BATCH, NUM_CLASSES)


@pytest.mark.unit
def test_mlp_head_multiple_hidden_layers_shape() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[1024, 512, 128])
    out = head(torch.randn(BATCH, EMBED_DIM))
    assert out.shape == (BATCH, NUM_CLASSES)


@pytest.mark.unit
def test_mlp_head_rejects_empty_hidden_dims() -> None:
    with pytest.raises(ValueError, match="at least one hidden layer"):
        MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[])


@pytest.mark.unit
def test_mlp_head_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[256], dropout=1.0)


@pytest.mark.unit
def test_mlp_head_zero_dropout_omits_dropout_layer() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[64], dropout=0.0)
    assert not any(isinstance(m, nn.Dropout) for m in head.net), (
        "dropout=0 should not insert a Dropout layer"
    )


@pytest.mark.unit
def test_mlp_head_positive_dropout_adds_dropout_layer() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[64], dropout=0.3)
    dropouts = [m for m in head.net if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0].p == pytest.approx(0.3)


@pytest.mark.unit
@pytest.mark.parametrize("norm,expected_layer", [
    ("batchnorm", nn.BatchNorm1d),
    ("layernorm", nn.LayerNorm),
])
def test_mlp_head_norm_inserts_correct_layer(norm, expected_layer) -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[64], norm=norm)
    assert any(isinstance(m, expected_layer) for m in head.net)


@pytest.mark.unit
def test_mlp_head_norm_none_inserts_no_norm() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[64], norm="none")
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)) for m in head.net)


# ---------- gradient flow ----------


@pytest.mark.unit
def test_mlp_head_gradient_flows_to_input() -> None:
    head = MLPHead(EMBED_DIM, NUM_CLASSES, hidden_dims=[256, 128], dropout=0.1)
    features = torch.randn(BATCH, EMBED_DIM, requires_grad=True)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    loss = nn.functional.cross_entropy(head(features), labels)
    loss.backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0


# ---------- activation/norm factory edge cases ----------


@pytest.mark.unit
def test_build_activation_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        _build_activation("swish")  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_norm_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown norm"):
        _build_norm("groupnorm", 32)  # type: ignore[arg-type]


# ---------- integration with LResNet50E_IR ----------


@pytest.mark.unit
def test_resnet_mlp_head_forward_shape() -> None:
    model = LResNet50E_IR(
        num_classes=NUM_CLASSES,
        head="mlp",
        pretrained=False,
        mlp_hidden_dims=[256, 128],
        mlp_activation="gelu",
        mlp_dropout=0.2,
        mlp_norm="layernorm",
    )
    # Skip the 25M-param backbone forward — feed embeddings directly.
    model.backbone = nn.Identity()
    out = model(torch.randn(BATCH, EMBED_DIM))
    assert out.shape == (BATCH, NUM_CLASSES)
    assert isinstance(model.head, MLPHead)


@pytest.mark.unit
def test_resnet_mlp_head_ignores_labels() -> None:
    """MLP head, like linear, must not change output when labels are passed."""
    model = LResNet50E_IR(
        num_classes=NUM_CLASSES,
        head="mlp",
        pretrained=False,
        mlp_hidden_dims=[64],
        mlp_dropout=0.0,
    )
    model.backbone = nn.Identity()
    model.dropout = nn.Identity()
    model.eval()

    features = torch.randn(BATCH, EMBED_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))

    out_no_label = model(features)
    out_with_label = model(features, labels)
    assert torch.allclose(out_no_label, out_with_label)


@pytest.mark.unit
def test_resnet_rejects_unknown_head() -> None:
    with pytest.raises(ValueError, match="head must be"):
        LResNet50E_IR(num_classes=NUM_CLASSES, head="cosface", pretrained=False)  # type: ignore[arg-type]


@pytest.mark.unit
def test_resnet_mlp_head_defaults_to_single_512_hidden() -> None:
    """When no mlp_hidden_dims is provided the head must still build (default=[512])."""
    model = LResNet50E_IR(num_classes=NUM_CLASSES, head="mlp", pretrained=False)
    assert isinstance(model.head, MLPHead)
    assert model.head.hidden_dims == [512]


# ---------- schema integration ----------


@pytest.mark.unit
def test_config_schema_accepts_mlp_head() -> None:
    from face_bias.config.schema import validate_config

    raw = {
        "model": {
            "head": "mlp",
            "mlp_hidden_dims": [1024, 256],
            "mlp_activation": "gelu",
            "mlp_dropout": 0.4,
            "mlp_norm": "batchnorm",
        }
    }
    normalised = validate_config(raw)
    assert normalised["model"]["head"] == "mlp"
    assert normalised["model"]["mlp_hidden_dims"] == [1024, 256]
    assert normalised["model"]["mlp_activation"] == "gelu"
    assert normalised["model"]["mlp_dropout"] == pytest.approx(0.4)
    assert normalised["model"]["mlp_norm"] == "batchnorm"


@pytest.mark.unit
def test_config_schema_rejects_nonpositive_hidden_dims() -> None:
    from pydantic import ValidationError

    from face_bias.config.schema import FaceBiasConfig

    with pytest.raises(ValidationError):
        FaceBiasConfig.model_validate({"model": {"head": "mlp", "mlp_hidden_dims": [512, 0]}})


@pytest.mark.unit
def test_config_schema_rejects_unknown_head() -> None:
    from pydantic import ValidationError

    from face_bias.config.schema import FaceBiasConfig

    with pytest.raises(ValidationError):
        FaceBiasConfig.model_validate({"model": {"head": "cosface"}})
