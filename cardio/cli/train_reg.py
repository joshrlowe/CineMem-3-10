"""Unified regression training CLI.

Generalises the per-dataset regression training scripts into one CLI
with a ``--dataset`` flag (read from ``config.data.name``).

Supported datasets (MVP): ``acdc``, ``mnms``, ``mnms2``.

Usage (Hydra)::

    cardio-train-reg --config-path /path/to --config-name config
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd

from cardio.data.datasets.classification import get_image_transforms
from cardio.data.datasets.regression import EndDiastoleEndSystoleDataset
from cardio.trainer.finetune import maybe_subset_dataset, run_train
from cardio.utils.logging import get_logger
from cardio.vision.classification import get_classification_or_regression_model
from cardio.vision.regression import regression_eval_dataloader, regression_loss

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset

logger = get_logger(__name__)

_SUPPORTED_DATASETS = {"acdc", "mnms", "mnms2"}


# =========================================================================
# Per-dataset load_dataset implementations
# =========================================================================


def _load_acdc(config: DictConfig) -> tuple[Dataset, Dataset]:
    data_dir = Path(config.data.dir).expanduser()
    meta_df = pd.read_csv(data_dir / "train_metadata.csv")
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    val_pids = meta_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
    train_meta_df = meta_df[~meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    val_meta_df = meta_df[meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train", meta_df=train_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train", meta_df=val_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=val_transform,
    )
    return train_dataset, val_dataset


def _load_mnms(config: DictConfig) -> tuple[Dataset, Dataset]:
    data_dir = Path(config.data.dir).expanduser()
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv")
    val_meta_df = pd.read_csv(data_dir / "val_metadata.csv")
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train", meta_df=train_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "val", meta_df=val_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=val_transform,
    )
    return train_dataset, val_dataset


def _load_mnms2(config: DictConfig) -> tuple[Dataset, Dataset]:
    data_dir = Path(config.data.dir).expanduser()
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv", dtype={"pid": str})
    val_meta_df = pd.read_csv(data_dir / "val_metadata.csv", dtype={"pid": str})
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train", meta_df=train_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "val", meta_df=val_meta_df,
        reg_col=reg_col, reg_mean=reg_mean, reg_std=reg_std,
        views=config.model.views, transform=val_transform,
    )
    return train_dataset, val_dataset


_LOADERS = {
    "acdc": _load_acdc,
    "mnms": _load_mnms,
    "mnms2": _load_mnms2,
}


# =========================================================================
# Entry point
# =========================================================================


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Train a regression model on a specified dataset."""
    dataset_name = config.data.name
    if dataset_name not in _LOADERS:
        msg = f"Unsupported dataset '{dataset_name}'. Choose from {_SUPPORTED_DATASETS}."
        raise ValueError(msg)

    load_dataset = _LOADERS[dataset_name]
    run_train(
        config=config,
        load_dataset=load_dataset,
        get_model_fn=get_classification_or_regression_model,
        loss_fn=regression_loss,
        eval_dataloader_fn=regression_eval_dataloader,
    )


if __name__ == "__main__":
    main()
