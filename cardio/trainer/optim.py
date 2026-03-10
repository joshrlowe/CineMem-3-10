"""Optimization utilities for CardioVLM training.

Ports the core scheduling, gradient scaling, checkpointing, and early
stopping utilities from CineMA, and adds:

- :class:`WarmupThenCosineScheduler` -- a ``LambdaLR``-based scheduler
  with linear warmup then half-cycle cosine decay.
- :func:`clip_grad_norm` -- thin wrapper around
  ``torch.nn.utils.clip_grad_norm_`` that returns the norm for logging.
"""

from __future__ import annotations

import math
from math import inf
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn, optim

from cardio.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = get_logger(__name__)


# =========================================================================
# Cosine scheduler (from CineMA / DINOv2)
# =========================================================================


class CosineScheduler:
    """Cosine scheduler with optional warmup and freeze phases.

    Builds a pre-computed schedule array of length *total_iters*.
    Index into it with ``scheduler[step]``.

    Reference: https://github.com/facebookresearch/dinov2
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ) -> None:
        """Initialise the cosine schedule array."""
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters,))
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        if len(self.schedule) != self.total_iters:
            msg = f"Schedule length {len(self.schedule)} != total_iters {self.total_iters}."
            raise ValueError(msg)

    def __getitem__(self, it: int) -> float | np.floating:
        """Return the scheduled value at iteration *it*."""
        if it >= self.total_iters:
            return self.final_value
        return self.schedule[it]


# =========================================================================
# Warmup-then-cosine (LambdaLR wrapper)
# =========================================================================


class WarmupThenCosineScheduler:
    """Linear warmup followed by half-cycle cosine decay, via ``LambdaLR``.

    Compatible with the standard ``optimizer.step()`` / ``scheduler.step()``
    pattern.

    Args:
        optimizer: wrapped optimizer.
        warmup_steps: number of linear-warmup steps.
        total_steps: total training steps (warmup included).
        min_lr_ratio: final LR as a fraction of the initial LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        """Initialise with warmup and cosine-decay parameters."""
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._lr_lambda
        )

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Return the last computed learning rate(s)."""
        return self._scheduler.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint."""
        self._scheduler.load_state_dict(state_dict)


# =========================================================================
# Gradient utilities
# =========================================================================


def get_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Compute the total gradient norm across *parameters*."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def clip_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_grad_norm: float = 1.0,
) -> torch.Tensor:
    """Clip gradient norm and return the (unclipped) total norm for logging.

    Thin wrapper around :func:`torch.nn.utils.clip_grad_norm_`.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)


# =========================================================================
# GradScaler
# =========================================================================


class GradScaler:
    """Gradient scaler with integrated gradient-norm clipping.

    Wraps :class:`torch.GradScaler` and provides a callable interface
    that performs ``backward`` + optional ``unscale`` + ``clip`` + ``step``
    + ``update`` in one call.
    """

    state_dict_key = "amp_scaler"

    def __init__(self) -> None:
        """Initialise the underlying ``torch.GradScaler``."""
        self._scaler = torch.GradScaler(
            "cuda", enabled=torch.cuda.is_available()
        )

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters: torch.Tensor | Iterable[torch.Tensor] | None = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> torch.Tensor | None:
        """Backward pass with gradient scaling.

        Args:
            loss: loss tensor (already divided by ``grad_accum_steps``
                if doing gradient accumulation).
            optimizer: optimizer to step.
            clip_grad: max gradient norm (``None`` = no clipping).
            parameters: model parameters (required when *update_grad*).
            create_graph: retain computation graph.
            update_grad: if ``False``, only accumulate gradients without
                stepping the optimizer or updating the scaler.

        Returns:
            Gradient norm (before clipping) when *update_grad* is True,
            else ``None``.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if parameters is None:
                msg = "parameters must not be None when update_grad=True."
                raise ValueError(msg)
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
            return norm
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return the scaler state dict."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scaler state from a dict."""
        self._scaler.load_state_dict(state_dict)


# =========================================================================
# Checkpointing
# =========================================================================


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model_wo_ddp: nn.Module,
    optimizer: optim.Optimizer,
    loss_scaler: GradScaler,
    n_samples: int,
    scheduler: WarmupThenCosineScheduler | None = None,
) -> Path:
    """Save a training checkpoint.

    Returns the path to the saved ``.pt`` file.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_{epoch}.pt"
    to_save: dict[str, Any] = {
        "model": model_wo_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "n_samples": n_samples,
    }
    if scheduler is not None:
        to_save["scheduler"] = scheduler.state_dict()
    torch.save(to_save, ckpt_path)
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    model_wo_ddp: nn.Module,
    optimizer: optim.Optimizer,
    loss_scaler: GradScaler,
    scheduler: WarmupThenCosineScheduler | None = None,
) -> tuple[nn.Module, optim.Optimizer, GradScaler, int, int]:
    """Load a checkpoint and restore model, optimizer, scaler, and scheduler.

    Returns:
        Tuple of ``(model, optimizer, loss_scaler, epoch, n_samples)``.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_wo_ddp.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    loss_scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Resumed from epoch {ckpt['epoch']}.")
    return model_wo_ddp, optimizer, loss_scaler, ckpt["epoch"], ckpt.get("n_samples", 0)


# =========================================================================
# Early stopping
# =========================================================================


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        min_delta: minimum decrease to qualify as improvement.
        patience: epochs without improvement before stopping.
    """

    def __init__(self, min_delta: float, patience: int) -> None:
        """Initialise early-stopping counters."""
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.patience = patience
        self.patience_count = 0
        self.should_stop = False
        self.has_improved = False

    def update(self, metric: float) -> None:
        """Update state after each epoch."""
        self.has_improved = self.best_metric > metric
        if self.has_improved and self.best_metric >= metric + self.min_delta:
            self.best_metric = metric
            self.patience_count = 0
        else:
            self.patience_count += 1
            self.should_stop = self.patience_count >= self.patience


# =========================================================================
# MAE / fine-tune learning-rate schedule
# =========================================================================


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: float,
    warmup_steps: int,
    max_n_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """Half-cycle cosine LR decay with linear warmup.

    Directly sets ``param_group["lr"]`` on the optimizer, respecting an
    optional ``lr_scale`` factor per group.

    Reference: https://github.com/facebookresearch/mae

    Args:
        optimizer: wrapped optimizer.
        step: current fractional step (e.g. ``batch_idx / len(loader) + epoch``).
        warmup_steps: number of warmup steps (epochs).
        max_n_steps: total number of steps (epochs).
        lr: peak learning rate.
        min_lr: minimum (final) learning rate.

    Returns:
        The computed learning rate for this step.
    """
    if step < warmup_steps:
        lr = lr * step / warmup_steps
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup_steps) / (max_n_steps - warmup_steps))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


# =========================================================================
# Gradient accumulation helper
# =========================================================================


def get_n_accum_steps(
    batch_size: int,
    batch_size_per_device: int,
    world_size: int,
) -> int:
    """Compute the number of gradient-accumulation steps.

    Args:
        batch_size: effective (logical) batch size.
        batch_size_per_device: micro-batch size per device.
        world_size: number of devices.

    Returns:
        Number of micro-steps before each optimizer step.

    Raises:
        ValueError: if sizes are incompatible.
    """
    batch_size_per_step = batch_size_per_device * world_size
    logger.info("batch_size_per_step = %d x %d = %d", batch_size_per_device, world_size, batch_size_per_step)
    if batch_size_per_step > batch_size:
        msg = f"batch_size_per_step {batch_size_per_step} > batch_size {batch_size}."
        raise ValueError(msg)
    if batch_size % batch_size_per_step != 0:
        msg = f"batch_size {batch_size} not divisible by batch_size_per_step {batch_size_per_step}."
        raise ValueError(msg)
    n_accum_steps = batch_size // batch_size_per_step
    if n_accum_steps > 1:
        logger.info("Gradient accumulation every %d steps (effective batch_size=%d).", n_accum_steps, batch_size)
    return n_accum_steps
