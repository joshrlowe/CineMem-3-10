"""Segmentation model training and evaluation utilities.

Ported from ``_reference/CineMA/cinema/segmentation/train.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cardio.data.transform import aggregate_patches, crop_start, get_patch_grid, patch_grid_sample
from cardio.eval.segmentation import segmentation_metrics
from cardio.utils.logging import get_logger
from cardio.vision.convunetr import get_model as _get_convunetr

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


# =========================================================================
# Model factory
# =========================================================================


def get_segmentation_model(config: DictConfig) -> nn.Module:
    """Return the segmentation model specified by *config*.

    Args:
        config: OmegaConf configuration.

    Returns:
        Instantiated segmentation model.
    """
    return _get_convunetr(config)


# =========================================================================
# Loss
# =========================================================================


def _segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss for one view (CE + Dice).

    Args:
        logits: ``(batch, n_classes, ...)``.
        labels: ``(batch, 1, ...)``.

    Returns:
        Tuple of ``(loss, metrics_dict)``.
    """
    labels = labels.long()
    mask = one_hot(labels.clamp(min=0), num_classes=logits.shape[1], dtype=logits.dtype)
    ce = F.cross_entropy(logits, labels.squeeze(dim=1), ignore_index=-1)
    dice = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)(logits, mask)
    loss = dice + ce
    metrics = {"cross_entropy": ce, "mean_dice_loss": dice, "loss": loss}
    return loss, metrics


def segmentation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
    loss_fn: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ] = _segmentation_loss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute segmentation loss across all views.

    Args:
        model: segmentation model.
        batch: batch dict with ``{view}_image`` and ``{view}_label`` keys.
        views: list of view names.
        device: target device.
        loss_fn: per-view loss callable.

    Returns:
        Tuple of ``(loss, metrics_dict)``.
    """
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
    label_dict = {view: batch[f"{view}_label"].to(device) for view in views}
    logits_dict = model(image_dict)

    metrics: dict[str, float] = {}
    losses: list[torch.Tensor] = []
    metric_keys: list[str] = []
    for view, logits in logits_dict.items():
        loss_view, metrics_view = loss_fn(logits, label_dict[view])
        metric_keys = list(metrics_view.keys())
        metrics_view[f"{view}_loss"] = loss_view
        losses.append(loss_view)
        metrics.update({f"{view}_{k}": v for k, v in metrics_view.items()})
    loss = sum(losses) / len(logits_dict)
    metrics["loss"] = loss
    metrics = {k: v.item() for k, v in metrics.items()}
    for k in metric_keys:
        metrics[k] = np.mean([metrics[f"{view}_{k}"] for view in logits_dict])
    return loss, metrics


# =========================================================================
# Inference / evaluation
# =========================================================================


def segmentation_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Forward pass with optional patch-based inference.

    For overlapped patches the logits are softmaxed to probabilities,
    averaged, and then ``log`` is taken.

    Args:
        model: segmentation model.
        image_dict: ``{view: (1, C, *spatial)}``.
        patch_size_dict: expected patch size per view.
        amp_dtype: AMP dtype.

    Returns:
        Dict of ``{view: (1, n_classes, *spatial)}``.
    """
    for view, image in image_dict.items():
        if any(s < p for s, p in zip(image.shape[2:], patch_size_dict[view], strict=False)):
            msg = f"View {view}: image {image.shape[2:]} < patch {patch_size_dict[view]}."
            raise ValueError(msg)

    views = list(image_dict.keys())
    need_patch_dict = {v: image_dict[v].shape[2:] != patch_size_dict[v] for v in views}

    if not any(need_patch_dict.values()):
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            return model(image_dict)

    if sum(need_patch_dict.values()) > 1:
        msg = f"Only one view may need patching, got {need_patch_dict}."
        raise ValueError(msg)
    batch_size = image_dict[views[0]].shape[0]
    if batch_size != 1:
        msg = f"Patch inference requires batch_size=1, got {batch_size}."
        raise ValueError(msg)

    view_to_patch = next(v for v, needs in need_patch_dict.items() if needs)
    image_to_patch = image_dict[view_to_patch][0]
    patch_size = patch_size_dict[view_to_patch]
    patch_overlap = tuple(s // 2 for s in patch_size)
    patch_start_indices = get_patch_grid(
        image_size=image_to_patch.shape[1:], patch_size=patch_size, patch_overlap=patch_overlap,
    )
    patches = patch_grid_sample(image_to_patch, patch_start_indices, patch_size)
    n_patches = patches.shape[0]

    logits_dict: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {v: patch if v == view_to_patch else image_dict[v] for v in views}
            patch_logits_dict = model(patch_image_dict)
            for v in views:
                logits_dict[v].append(patch_logits_dict[v])

    aggregated: dict[str, torch.Tensor] = {}
    for v in views:
        logits = torch.cat(logits_dict[v], dim=0)
        if v == view_to_patch:
            probs = F.softmax(logits, dim=1)
            probs = aggregate_patches(probs, start_indices=patch_start_indices, image_size=image_to_patch.shape[1:])
            logits = torch.log(probs)
        else:
            logits = torch.log(torch.mean(F.softmax(logits, dim=1), dim=0))
        aggregated[v] = logits[None, ...]
    return aggregated


def segmentation_eval(  # noqa: C901
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics_fn: Callable[
        [torch.Tensor, torch.Tensor, tuple[float, ...]],
        dict[str, torch.Tensor],
    ] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Evaluate one batch.

    Args:
        model: segmentation model.
        batch: data batch (batch_size=1 for patching).
        patch_size_dict: patch sizes per view.
        spacing_dict: spacings per view.
        amp_dtype: AMP dtype.
        device: target device.
        metrics_fn: per-view metrics callable (``None`` skips metric computation).

    Returns:
        Tuple of ``(logits_dict, metrics)``.
    """
    views = list(patch_size_dict.keys())
    image_dict = {v: batch[f"{v}_image"].to(device) for v in views}
    logits_dict = segmentation_forward(model, image_dict, patch_size_dict, amp_dtype)

    for v in views:
        width = int(batch[f"{v}_width"][0])
        height = int(batch[f"{v}_height"][0])
        if len(patch_size_dict[v]) == 3:
            n_slices = int(batch["n_slices"][0])
            logits_dict[v] = crop_start(logits_dict[v], (*logits_dict[v].shape[:2], width, height, n_slices))
        elif len(patch_size_dict[v]) == 2:
            logits_dict[v] = crop_start(logits_dict[v], (*logits_dict[v].shape[:2], width, height))
        else:
            msg = f"Invalid patch size dimensionality: {patch_size_dict[v]}."
            raise ValueError(msg)

    if metrics_fn is None:
        return logits_dict, {}

    label_dict = {v: batch[f"{v}_label"].to(device) for v in views}
    for v in views:
        width = int(batch[f"{v}_width"][0])
        height = int(batch[f"{v}_height"][0])
        if len(patch_size_dict[v]) == 3:
            n_slices = int(batch["n_slices"][0])
            label_dict[v] = crop_start(label_dict[v], (*label_dict[v].shape[:2], width, height, n_slices))
        elif len(patch_size_dict[v]) == 2:
            label_dict[v] = crop_start(label_dict[v], (*label_dict[v].shape[:2], width, height))
        else:
            msg = f"Invalid patch size dimensionality: {patch_size_dict[v]}."
            raise ValueError(msg)

    metrics: dict[str, float] = {}
    metric_keys: list[str] = []
    for v in views:
        mv = metrics_fn(logits_dict[v], label_dict[v], spacing_dict[v])
        metric_keys = list(mv.keys())
        for k, val in mv.items():
            metrics[f"{v}_{k}"] = float(val.cpu().to(dtype=torch.float32).numpy())
    for k in metric_keys:
        metrics[k] = np.mean([metrics[f"{v}_{k}"] for v in views])
    return logits_dict, metrics


def segmentation_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics_fn: Callable[
        [torch.Tensor, torch.Tensor, tuple[float, ...]],
        dict[str, torch.Tensor],
    ] = segmentation_metrics,
) -> dict[str, float]:
    """Evaluate segmentation model over a full dataloader.

    Args:
        model: segmentation model.
        dataloader: validation data loader.
        patch_size_dict: patch sizes per view.
        spacing_dict: spacings per view.
        amp_dtype: AMP dtype.
        device: target device.
        metrics_fn: per-sample metrics callable.

    Returns:
        Mean metrics over the dataset.
    """
    metrics: dict[str, list[float]] = defaultdict(list)
    for _, batch in enumerate(dataloader):
        _, sample_metrics = segmentation_eval(
            model, batch, patch_size_dict, spacing_dict, amp_dtype, device, metrics_fn,
        )
        for k, v in sample_metrics.items():
            metrics[k].append(v)
    return {k: np.nanmean(v) for k, v in metrics.items()}
