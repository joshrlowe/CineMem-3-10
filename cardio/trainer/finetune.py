"""Fine-tuning training loop for downstream vision tasks.

Ported from ``_reference/CineMA/cinema/train.py``.  Provides a generic
:func:`run_train` entry-point that is shared by segmentation,
classification, and regression CLI scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cardio.trainer.optim import (
    EarlyStopping,
    GradScaler,
    adjust_learning_rate,
    get_n_accum_steps,
    save_checkpoint,
)
from cardio.utils.device import get_amp_dtype_and_device, print_model_info
from cardio.utils.logging import get_logger, init_wandb
from cardio.vision.convvit import load_pretrain_weights, param_groups_lr_decay

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from omegaconf import DictConfig

logger = get_logger(__name__)


# =========================================================================
# Dataset helpers
# =========================================================================


def maybe_reduce_batch_size(config: DictConfig, n: int) -> DictConfig:
    """Halve batch size until it fits within the dataset.

    Args:
        config: OmegaConf configuration (mutated in-place).
        n: number of training samples.

    Returns:
        The (potentially modified) config.
    """
    batch_size = config.train.batch_size
    if n >= batch_size:
        return config
    while n < batch_size:
        batch_size //= 2
    if batch_size == 0:
        msg = f"Dataset size too small ({n})."
        raise ValueError(msg)
    logger.warning("Using batch size %d instead of %d.", batch_size, config.train.batch_size)
    config.train.batch_size = batch_size
    config.train.batch_size_per_device = min(config.train.batch_size_per_device, batch_size)
    return config


def maybe_subset_dataset(
    config: DictConfig,
    train_meta_df: pd.DataFrame,
    val_meta_df: pd.DataFrame,
    group_col: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally sub-sample the train/val DataFrames.

    Args:
        config: OmegaConf configuration.
        train_meta_df: training metadata.
        val_meta_df: validation metadata.
        group_col: column for stratified sampling (empty = random).

    Returns:
        Tuple of ``(train_meta_df, val_meta_df)``.
    """
    if config.data.max_n_samples > 0:
        train_ratio = min(config.data.max_n_samples / len(train_meta_df), 1.0)
        val_ratio = min(config.data.max_n_samples / len(val_meta_df), 1.0)
        if group_col:
            train_meta_df = train_meta_df.groupby(group_col).sample(frac=train_ratio, random_state=0)
            val_meta_df = val_meta_df.groupby(group_col).sample(frac=train_ratio, random_state=0)
        else:
            train_meta_df = train_meta_df.sample(frac=train_ratio, random_state=0, ignore_index=True)
            val_meta_df = val_meta_df.sample(frac=val_ratio, random_state=0, ignore_index=True)
        logger.info(
            "Using %d samples for training and %d for validation.",
            train_meta_df.shape[0],
            val_meta_df.shape[0],
        )
    if config.data.proportion < 1:
        train_meta_df = train_meta_df.sample(
            n=int(config.data.proportion * len(train_meta_df)),
            random_state=config.seed,
            ignore_index=True,
        )
        logger.info(
            "Using %.0f%% samples, %d samples for training.",
            config.data.proportion * 100,
            train_meta_df.shape[0],
        )
    return train_meta_df, val_meta_df


# =========================================================================
# Single-epoch loop
# =========================================================================


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    amp_dtype: torch.dtype,
    device: torch.device,
    epoch: int,
    n_accum_steps: int,
    n_samples: int,
    config: DictConfig,
    wandb_run: object | None,
    loss_fn: Callable[
        [nn.Module, dict[str, torch.Tensor], list[str], torch.device],
        tuple[torch.Tensor, dict[str, float]],
    ],
) -> int:
    """Fine-tune *model* for one epoch.

    Args:
        model: model to train.
        train_dataloader: training data loader.
        optimizer: optimizer.
        loss_scaler: :class:`GradScaler` for mixed-precision training.
        amp_dtype: dtype for ``torch.autocast``.
        device: target device.
        epoch: current epoch index.
        n_accum_steps: gradient-accumulation steps.
        n_samples: running sample counter.
        config: OmegaConf configuration.
        wandb_run: optional W&B run for logging.
        loss_fn: callable ``(model, batch, views, device) -> (loss, metrics)``.

    Returns:
        Updated sample counter.
    """
    model.train()
    views = train_dataloader.dataset.views
    batch_size_per_step = config.train.batch_size_per_device
    clip_grad = config.train.clip_grad if config.train.clip_grad > 0 else None

    for i, batch in enumerate(train_dataloader):
        lr = adjust_learning_rate(
            optimizer=optimizer,
            step=i / len(train_dataloader) + epoch,
            warmup_steps=config.train.n_warmup_epochs,
            max_n_steps=config.train.n_epochs,
            lr=config.train.lr,
            min_lr=config.train.min_lr,
        )
        with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            loss, metrics = loss_fn(model, batch, views, device)
            metrics = {f"train_{k}": v for k, v in metrics.items()}

        if torch.isnan(loss).any():
            logger.error("Got NaN loss, metrics are %s.", metrics)
            continue

        loss /= n_accum_steps
        update_grad = (i + 1) % n_accum_steps == 0
        grad_norm = loss_scaler(
            loss=loss,
            optimizer=optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            update_grad=update_grad,
        )
        if update_grad:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_samples += batch_size_per_step
        if update_grad and wandb_run is not None:
            metrics.update({
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "n_samples": n_samples,
                "epoch": epoch,
            })
            wandb_run.log(metrics)

    return n_samples


# =========================================================================
# Main fine-tuning driver
# =========================================================================


def run_train(  # noqa: C901
    config: DictConfig,
    load_dataset: Callable[[DictConfig], tuple[Dataset, Dataset]],
    get_model_fn: Callable[[DictConfig], nn.Module],
    loss_fn: Callable[
        [nn.Module, dict[str, torch.Tensor], list[str], torch.device],
        tuple[torch.Tensor, dict[str, float]],
    ],
    eval_dataloader_fn: Callable[
        [nn.Module, DataLoader, dict, dict, torch.dtype, torch.device],
        dict[str, float],
    ],
) -> None:
    """Orchestrate fine-tuning for any downstream vision task.

    Args:
        config: OmegaConf configuration.
        load_dataset: ``(config) -> (train_dataset, val_dataset)``.
        get_model_fn: ``(config) -> model``.
        loss_fn: ``(model, batch, views, device) -> (loss, metrics)``.
        eval_dataloader_fn: ``(model, dl, patch_sizes, spacings, dtype, device) -> metrics``.
    """
    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)

    # ----- dataset --------------------------------------------------------
    train_dataset, val_dataset = load_dataset(config=config)
    config = maybe_reduce_batch_size(config, len(train_dataset))

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.train.batch_size_per_device,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,
    )

    # ----- model ----------------------------------------------------------
    views = [config.model.views] if isinstance(config.model.views, str) else list(config.model.views)
    model = get_model_fn(config)
    print_model_info(model)
    if config.model.ckpt_path is not None:
        model = load_pretrain_weights(
            model=model,
            views=views,
            ckpt_path=Path(config.model.ckpt_path),
            freeze=config.model.freeze_pretrained,
        )
    model.to(device)

    # ----- W&B ------------------------------------------------------------
    tags = [
        config.data.name,
        config.model.name,
        *views,
        config.task,
        f"seed{config.seed}",
        f"{int(config.data.proportion * 100)}%",
    ]
    if config.model.ckpt_path is not None:
        tags.append("finetuned")
    if hasattr(config.data, "class_column"):
        tags.append(config.data.class_column)
    if hasattr(config.data, "regression_column"):
        tags.append(config.data.regression_column)
    wandb_run, ckpt_dir = init_wandb(config=config, tags=sorted(set(tags)))

    # ----- optimizer ------------------------------------------------------
    logger.info("Initializing optimizer.")
    if config.model.ckpt_path is not None:
        param_groups = param_groups_lr_decay(
            model,
            no_weight_decay_list=[],
            weight_decay=config.train.weight_decay,
            layer_decay=config.train.layer_decay,
        )
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # ----- training loop --------------------------------------------------
    logger.info("Start training.")
    patch_size_dict = {
        v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size
        for v in views
    }
    spacing_dict = {
        v: config.data.sax.spacing if v == "sax" else config.data.lax.spacing
        for v in views
    }
    early_stop = EarlyStopping(
        min_delta=config.train.early_stopping.min_delta,
        patience=config.train.early_stopping.patience,
    )
    n_samples = 0
    saved_ckpt_paths: list[Path] = []

    for epoch in range(config.train.n_epochs):
        optimizer.zero_grad()
        n_samples = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            amp_dtype=amp_dtype,
            device=device,
            epoch=epoch,
            n_accum_steps=n_accum_steps,
            n_samples=n_samples,
            config=config,
            wandb_run=wandb_run,
            loss_fn=loss_fn,
        )

        if ckpt_dir is None or (epoch + 1) % config.train.eval_interval != 0:
            continue

        # --- evaluation ---------------------------------------------------
        logger.info("Start evaluating model at epoch %d.", epoch)
        model.eval()

        val_metrics = eval_dataloader_fn(
            model, val_dataloader, patch_size_dict, spacing_dict, amp_dtype, device,
        )
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        val_metrics["n_samples"] = n_samples
        if wandb_run is not None:
            wandb_run.log(val_metrics)
        val_metrics_str = {
            k: v if isinstance(v, int) else f"{v:.2e}" for k, v in val_metrics.items()
        }
        logger.info("Validation metrics: %s.", val_metrics_str)

        # --- early stopping -----------------------------------------------
        early_stop_metric = val_metrics[config.train.early_stopping.metric]
        if config.train.early_stopping.mode == "max":
            early_stop_metric = -early_stop_metric
        early_stop.update(early_stop_metric)
        logger.info(
            "Early stop updated %d: should_stop=%s, patience_count=%d, patience=%d.",
            epoch,
            early_stop.should_stop,
            early_stop.patience_count,
            early_stop.patience,
        )

        # --- checkpoint ---------------------------------------------------
        if early_stop.has_improved or epoch == 0:
            ckpt_path = save_checkpoint(ckpt_dir, epoch, model, optimizer, loss_scaler, n_samples)
            saved_ckpt_paths.append(ckpt_path)
            logger.info("Saved checkpoint epoch %d at %s (%d samples).", epoch, ckpt_path, n_samples)
            if len(saved_ckpt_paths) > config.train.max_n_ckpts > 0:
                to_delete = saved_ckpt_paths.pop(0)
                to_delete.unlink(missing_ok=True)
                logger.info("Deleted outdated checkpoint %s.", to_delete)

        if early_stop.should_stop:
            logger.info(
                "Early stopping: %s = %s, patience %d.",
                config.train.early_stopping.metric,
                early_stop.best_metric,
                early_stop.patience_count,
            )
            break

    if saved_ckpt_paths:
        logger.info("Last checkpoint: %s", saved_ckpt_paths[-1])
