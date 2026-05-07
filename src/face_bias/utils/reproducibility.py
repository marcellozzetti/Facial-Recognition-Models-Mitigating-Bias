"""Determinism helpers — call ``seed_everything(seed)`` at the top of every entrypoint."""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True, warn_only: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs so an experiment is reproducible.

    Args:
        seed: Integer seed shared across all RNGs.
        deterministic: When True, set cuDNN to deterministic mode and ask
            PyTorch to fail on non-deterministic ops. Slower but reproducible.
        warn_only: Forward to ``torch.use_deterministic_algorithms``; when
            True, emit a warning instead of raising for ops without a
            deterministic implementation. Defaults to True so existing
            torchvision pipelines (which use a few non-deterministic kernels)
            still run.

    Returns:
        The seed used (handy for callers that pass it on to MLflow params).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash randomness used by Python's set/dict iteration order.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS workspace must be set before the first CUDA op for full
    # determinism on Ampere+. See https://docs.nvidia.com/cuda/cublas/index.html
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.benchmark = True

    logging.info(
        f"Seeded RNGs (seed={seed}, deterministic={deterministic}, "
        f"cuda_available={torch.cuda.is_available()})"
    )
    return seed


def seed_from_config(config: dict, key: str = "random_state") -> Optional[int]:
    """Read the seed from ``config['training'][key]`` and apply it."""
    seed = config.get("training", {}).get(key)
    if seed is None:
        logging.warning(f"No training.{key} in config; runs will not be reproducible.")
        return None
    return seed_everything(int(seed))
