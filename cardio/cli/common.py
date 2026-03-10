"""Shared CLI helpers for CardioVLM.

Re-exports common utilities and adds a Hydra configuration loader
for vision task scripts that need OmegaConf ``DictConfig`` objects.
"""

from __future__ import annotations

from cardio.utils.misc import load_yaml
from cardio.vlm.config import build_cinemem_config

__all__ = [
    "build_cinemem_config",
    "load_hydra_config",
    "load_yaml",
]


def load_hydra_config(config_path: str) -> object:
    """Load an OmegaConf ``DictConfig`` from a YAML file.

    This is a lightweight alternative to ``@hydra.main`` for scripts
    that accept a ``--config`` path at the CLI level rather than relying
    on Hydra's automatic config composition.

    Args:
        config_path: path to a YAML configuration file.

    Returns:
        An OmegaConf ``DictConfig`` instance.
    """
    from omegaconf import OmegaConf

    return OmegaConf.load(config_path)
