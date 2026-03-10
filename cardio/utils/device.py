"""Distributed training and device utilities.

References:
    https://github.com/facebookresearch/mae/blob/main/util/misc.py
    https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

from __future__ import annotations

import datetime
import os
import socket
from contextlib import closing

import torch
from torch import nn
from torch.backends import cudnn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from cardio.utils.logging import get_logger

logger = get_logger(__name__)


def get_free_port() -> int:
    """Return a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ddp_setup(rank: int, world_size: int, port: int) -> None:
    """Initialise a NCCL process group for distributed data-parallel training.

    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
        port: Port number for the master process.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    logger.info("Setting MASTER_ADDR=localhost.")
    logger.info(f"Setting MASTER_PORT={port}.")
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(rank)


def get_amp_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    """Detect the best AMP dtype and device for the current hardware.

    Returns:
        Tuple of (amp_dtype, device).
    """
    amp_dtype = torch.float16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        cudnn.benchmark = True
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("Using bfloat16 for automatic mixed precision.")
    else:
        logger.info("CUDA is not available, using CPU.")
        device = torch.device("cpu")

    return amp_dtype, device


def print_model_info(model: nn.Module) -> None:
    """Log total and trainable parameter counts for *model*."""
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_trainable_params:,}")


def setup_ddp_model(model: nn.Module, device: torch.device, rank: int, world_size: int) -> tuple[nn.Module, nn.Module]:
    """Wrap *model* in DDP when ``world_size > 1``.

    Args:
        model: Model to setup.
        device: Device to use.
        rank: Unique identifier of each process.
        world_size: Total number of processes.

    Returns:
        Tuple of (model, model_without_ddp).
    """
    model.to(device)
    model_wo_ddp = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
        model_wo_ddp = model.module
    return model, model_wo_ddp


def get_device_map(model_name: str) -> dict:  # noqa: ARG001
    """Return a ``device_map`` kwarg dict suitable for ``from_pretrained``.

    Accelerate's ``device_map="auto"`` creates meta-tensor placeholders that
    break on MPS, so we fall back to ``None`` there and move manually.

    Args:
        model_name: HuggingFace model identifier (reserved for future
            per-model overrides).

    Returns:
        Dict with a ``"device_map"`` key ready to be unpacked into
        ``AutoModel.from_pretrained(**get_device_map(name))``.
    """
    if torch.cuda.is_available():
        return {"device_map": "auto"}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"device_map": None}
    return {"device_map": "cpu"}
