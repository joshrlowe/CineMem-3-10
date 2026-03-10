"""Logging and experiment-tracking utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a rich StreamHandler attached.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level (default ``INFO``).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = "%(threadName)s-%(process)s | %(asctime)s-%(name)s-%(funcName)s:%(lineno)d | [%(levelname)s] %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    """Flatten a nested dict into a single-level dict with joined keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(d=v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def init_wandb(
    config: dict[str, Any] | object,
    tags: list[str] | None = None,
    project: str | None = None,
    run_name: str | None = None,
) -> tuple[Any, Path]:
    """Initialise a Weights & Biases run and resolve a checkpoint directory.

    Supports two calling conventions:

    * **Vision (Hydra/OmegaConf)**: ``config`` is a ``DictConfig`` with
      ``config.logging.wandb.project`` and optional ``config.logging.dir``.
    * **VLM (argparse)**: ``project`` is passed explicitly as a kwarg;
      ``config`` is a plain dict of hyperparameters.

    Args:
        config: Configuration dict (or OmegaConf ``DictConfig``).
        tags: Optional W&B run tags.
        project: Explicit project name (overrides any value in *config*).
        run_name: Optional display name for the run.

    Returns:
        Tuple of ``(wandb_run | None, ckpt_dir)``.
    """
    _is_omegaconf = hasattr(config, "_metadata")

    if project is None and _is_omegaconf:
        project = config.logging.wandb.project

    if not project:
        log_dir = None
        if _is_omegaconf and hasattr(config, "logging") and config.logging.dir is not None:
            log_dir = config.logging.dir
        ckpt_dir = Path(log_dir) / "ckpt" if log_dir else Path("ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return None, ckpt_dir

    import wandb  # lazy import

    entity = None
    if _is_omegaconf and hasattr(config, "logging"):
        entity = getattr(config.logging.wandb, "entity", None)

    flat = flatten_dict(dict(config)) if config else {}
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        config=flat,
        tags=tags,
        name=run_name,
    )

    log_dir = None
    if _is_omegaconf and hasattr(config, "logging"):
        log_dir = getattr(config.logging, "dir", None)

    if log_dir is None:
        ckpt_dir = Path(wandb_run.settings.files_dir).parent / "ckpt"
    else:
        ckpt_dir = Path(log_dir) / Path(wandb_run.settings.files_dir).parent.name / "ckpt"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _is_omegaconf:
        from omegaconf import OmegaConf

        OmegaConf.save(config=config, f=ckpt_dir / "config.yaml")

    return wandb_run, ckpt_dir
