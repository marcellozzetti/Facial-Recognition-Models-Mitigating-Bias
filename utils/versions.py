import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import platform
import sys
import importlib
import logging
from utils.config import load_config
from utils.custom_logging import setup_logging

def get_system_info():
    """
    Collect system information, including the operating system and Python version.
    """
    return {
        "Python Version": sys.version,
        "OS": f"{platform.system()} {platform.release()} ({platform.version()})",
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Node Name": platform.node()
    }

def get_library_version(library_name):
    """
    Attempt to import a library and retrieve its version.
    """
    try:
        lib = importlib.import_module(library_name)
        return getattr(lib, '__version__', 'Unknown version')
    except ImportError:
        return 'Not installed'

def system_info_report(additional_libraries=None):
    """
    Generate a report with system information and specified library versions.
    """
    if additional_libraries is None:
        additional_libraries = []

    # Collect system information
    report = get_system_info()

    # Check critical and additional libraries
    critical_libraries = ['torch']
    for lib in critical_libraries + additional_libraries:
        report[f"{lib.capitalize()} Version"] = get_library_version(lib)

    # Log the report
    logging.info("=== System and Library Information ===")
    for key, value in sorted(report.items()):
        logging.info(f"{key}: {value}")

    return report

# Usage example
if __name__ == "__main__":
    config = load_config('configs/default.yaml')
    setup_logging(config, 'log_version_file')
    
    # Additional libraries to check
    additional_libs = ['numpy', 'pandas']
    system_info_report(additional_libs)