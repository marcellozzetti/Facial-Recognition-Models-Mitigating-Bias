"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml

from face_bias.config.schema import validate_config


def load_config(config_path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: path to the YAML file.
        validate: when True (default), the raw dict is validated against
            ``FaceBiasConfig`` and a normalised dict is returned. Set to
            False to skip validation (e.g. when intentionally loading a
            partial fixture).
    """
    with open(config_path) as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    if validate:
        return validate_config(raw)
    return raw
