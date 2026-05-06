"""Environment introspection — Python version, OS, library versions, GPU."""

import importlib
import logging
import platform
import sys
from typing import Any

import torch


def get_system_info() -> dict[str, str]:
    """Collect system information."""
    return {
        "Python Version": sys.version,
        "OS": f"{platform.system()} {platform.release()} ({platform.version()})",
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Node Name": platform.node(),
    }


def get_library_version(library_name: str) -> str:
    """Return the version string of an installed library, or 'Not installed'."""
    try:
        lib = importlib.import_module(library_name)
        return getattr(lib, "__version__", "Unknown version")
    except ImportError:
        return "Not installed"


def system_info_report(additional_libraries: list[str] | None = None) -> dict[str, Any]:
    """Generate a report with system information and specified library versions."""
    if additional_libraries is None:
        additional_libraries = []

    report: dict[str, Any] = dict(get_system_info())

    critical_libraries = ["torch"]
    for lib in critical_libraries + additional_libraries:
        report[f"{lib.capitalize()} Version"] = get_library_version(lib)

    logging.info("=== System and Library Information ===")
    for key, value in sorted(report.items()):
        logging.info(f"{key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Device: {device}")

    return report
