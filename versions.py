import platform
import torch
import sys
import os

# Versão do Python
print(f"Python Version: {sys.version}")

# Versão do PyTorch
print(f"PyTorch Version: {torch.__version__}")

# Versão do sistema operacional
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")

# Informações sobre a arquitetura da máquina
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Node Name: {platform.node()}")

# Outras bibliotecas que você quiser verificar
try:
    import numpy as np
    print(f"Numpy Version: {np.__version__}")
except ImportError:
    print("Numpy not installed")

try:
    import pandas as pd
    print(f"Pandas Version: {pd.__version__}")
except ImportError:
    print("Pandas not installed")

try:
    import torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
except ImportError:
    print("Torchvision not installed")
