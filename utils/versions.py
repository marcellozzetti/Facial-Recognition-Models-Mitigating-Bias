import platform
import torch
import sys
import os

def system_info_report():
    """
    Generates a report of system and library versions, ensuring critical dependencies are verified.
    """
    print("=== System and Library Information ===\n")

    # Python Version
    print(f"Python Version: {sys.version}")

    # PyTorch Version
    print(f"PyTorch Version: {torch.__version__}")

    # Operating System Version
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")

    # Machine Architecture
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Node Name: {platform.node()}")

    # Additional Library Versions
    def check_library(name, import_statement, version_attr):
        try:
            lib = __import__(import_statement)
            version = getattr(lib, version_attr, "Unknown")
            print(f"{name} Version: {version}")
        except ImportError:
            print(f"{name} not installed")

    check_library("Numpy", "numpy", "__version__")
    check_library("Pandas", "pandas", "__version__")
    check_library("Torchvision", "torchvision", "__version__")
