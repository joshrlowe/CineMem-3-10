"""Miscellaneous helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch_dtype(name: str) -> torch.dtype:
    """Convert a human-readable dtype string to a :class:`torch.dtype`.

    Accepted values: ``"fp16"``, ``"float16"``, ``"half"``, ``"bf16"``,
    ``"bfloat16"``, ``"fp32"``, ``"float32"``, ``"full"``.

    Raises:
        ValueError: If *name* is not recognised.
    """
    name = str(name).lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "full"):
        return torch.float32
    msg = f"Unknown dtype: {name}"
    raise ValueError(msg)


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist, then return it."""
    os.makedirs(path, exist_ok=True)
    return path


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file and return the parsed dict."""
    import yaml

    with open(path, encoding="utf-8") as f:  # noqa: PTH123
        return yaml.safe_load(f)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (optionally trainable) parameters in *model*."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
