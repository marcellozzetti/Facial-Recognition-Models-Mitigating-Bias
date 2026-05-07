"""Tests for face_bias.utils.logging structured-logging helpers."""

import logging
import re
from pathlib import Path

import pytest

from face_bias.utils.logging import RunIdFilter, make_run_id, setup_logging


def _basic_config(log_dir: Path) -> dict:
    return {
        "logging": {
            "log_dir": str(log_dir),
            "log_level": "INFO",
            "format": "%(asctime)s [%(run_id)s] %(message)s",
            "test_log": "test.log",
        }
    }


@pytest.mark.unit
def test_make_run_id_format() -> None:
    rid = make_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", rid), rid


@pytest.mark.unit
def test_run_ids_are_unique() -> None:
    assert make_run_id() != make_run_id()


@pytest.mark.unit
def test_runidfilter_injects_attribute() -> None:
    f = RunIdFilter("abc123")
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    assert f.filter(record) is True
    assert record.run_id == "abc123"


@pytest.mark.unit
def test_setup_logging_returns_run_id_and_writes_file(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)

    rid = setup_logging(config, "test_log", run_id="20260506T120000Z-deadbe")

    assert rid == "20260506T120000Z-deadbe"
    log_file = tmp_path / "test.log"
    assert log_file.exists()

    logging.info("hello world")

    # Force handlers to flush before reading.
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log_file.read_text(encoding="utf-8")
    assert "[20260506T120000Z-deadbe]" in text
    assert "hello world" in text


@pytest.mark.unit
def test_setup_logging_generates_run_id_when_omitted(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)
    rid = setup_logging(config, "test_log")
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", rid)


@pytest.mark.unit
def test_setup_logging_replaces_previous_handlers(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)
    setup_logging(config, "test_log", run_id="run1")
    setup_logging(config, "test_log", run_id="run2")
    handlers = logging.getLogger().handlers
    # Two handlers (file + stream) for the second run only — first run's
    # handlers must have been removed.
    assert len(handlers) == 2
