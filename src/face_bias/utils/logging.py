"""Structured logging with a per-run correlation id.

Every CLI invocation generates a short ``run_id`` (UTC timestamp + 6 hex
chars) that is injected into every log record via a logging filter. The
id is also returned to the caller so it can be passed to MLflow as a
tag/parameter, making the logs and the experiment record cross-linkable.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
from datetime import datetime, timezone
from typing import Any

DEFAULT_FORMAT = "%(asctime)s [%(run_id)s] %(name)s %(levelname)s: %(message)s"


class RunIdFilter(logging.Filter):
    """Attach a fixed ``run_id`` attribute to every log record."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def make_run_id() -> str:
    """Generate a new run id like ``20260506T172359Z-a1b2c3``."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def setup_logging(
    config: dict[str, Any],
    log_file_key: str,
    *,
    run_id: str | None = None,
) -> str:
    """Configure root logger with structured format + run_id tagging.

    The previous signature ``setup_logging(config, log_file_key)`` is
    preserved (run_id is optional). The function now returns the
    ``run_id`` string so callers (CLI entry-points) can stash it in
    MLflow or surface it to the user.
    """
    if run_id is None:
        run_id = make_run_id()

    log_dir = config["logging"]["log_dir"]
    log_file = os.path.join(log_dir, config["logging"][log_file_key])
    log_level = config["logging"].get("log_level", "INFO")
    log_format = config["logging"].get("format", DEFAULT_FORMAT)
    if "%(run_id)s" not in log_format:
        # Caller customised format but did not opt out of run_id; prepend it.
        log_format = f"[%(run_id)s] {log_format}"

    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stderr)
    run_filter = RunIdFilter(run_id)
    for handler in (file_handler, stream_handler):
        handler.addFilter(run_filter)
        handler.setFormatter(logging.Formatter(log_format))

    root = logging.getLogger()
    # Replace any previously installed handlers (e.g. when running multiple
    # CLIs from the same Python process during tests).
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.info(f"Run started - log file: {log_file}")
    return run_id
