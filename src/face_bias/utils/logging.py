"""Logging setup helpers."""

import logging
import os
from typing import Any


def setup_logging(config: dict[str, Any], log_file_key: str) -> None:
    """Configure root logger to write to both stderr and a per-stage file."""
    log_dir = config["logging"]["log_dir"]
    log_file = os.path.join(log_dir, config["logging"][log_file_key])

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=config["logging"]["log_level"],
        format=config["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
