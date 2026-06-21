"""Reproducible seeding across Python, NumPy and (optionally) PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    """Seed every RNG we can reach for reproducible experiments.

    Parameters
    ----------
    seed:
        The base random seed.
    deterministic_torch:
        If ``True`` and PyTorch is installed, enable deterministic cuDNN
        algorithms (slower, but bit-for-bit reproducible on the same hardware).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional for the lightweight parts of the codebase
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch genuinely optional
        pass


__all__ = ["set_global_seeds"]
