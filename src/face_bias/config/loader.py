"""YAML configuration loader.

Pydantic schema validation will be added in Sprint B (B6).
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)
