"""Tests for face_bias.config.schema validation."""

import pytest
from pydantic import ValidationError

from face_bias.config import FaceBiasConfig, load_config, validate_config


@pytest.mark.unit
def test_default_yaml_validates_clean() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg["model"]["num_classes"] == 7
    assert cfg["image"]["image_std"] != cfg["image"]["image_mean"]


@pytest.mark.unit
def test_minimal_dict_fills_defaults() -> None:
    cfg = validate_config({})
    assert cfg["model"]["head"] == "linear"
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["image"]["image_size"] == (224, 224)


@pytest.mark.unit
def test_invalid_head_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({"model": {"head": "transformer"}})


@pytest.mark.unit
def test_invalid_optimizer_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({"training": {"optimizer": "rmsprop"}})


@pytest.mark.unit
def test_invalid_dropout_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({"model": {"dropout": 1.5}})


@pytest.mark.unit
def test_invalid_test_size_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({"training": {"test_size": 1.0}})


@pytest.mark.unit
def test_log_level_normalised_to_upper() -> None:
    cfg = validate_config({"logging": {"log_level": "info"}})
    assert cfg["logging"]["log_level"] == "INFO"


@pytest.mark.unit
def test_unknown_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_config({"logging": {"log_level": "VERBOSE"}})


@pytest.mark.unit
def test_extra_keys_are_ignored() -> None:
    cfg = validate_config({"model": {"name": "X", "future_field": 123}})
    assert cfg["model"]["name"] == "X"
    assert "future_field" not in cfg["model"]


@pytest.mark.unit
def test_facebiasconfig_directly() -> None:
    parsed = FaceBiasConfig.model_validate({})
    assert parsed.model.num_classes == 7
    assert parsed.image.image_size == (224, 224)


@pytest.mark.unit
def test_load_config_skip_validation() -> None:
    cfg = load_config("configs/default.yaml", validate=False)
    # Without validation we still get the raw YAML content.
    assert "model" in cfg
