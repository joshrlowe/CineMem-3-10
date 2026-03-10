"""Regression model training and evaluation utilities.

Ported from ``_reference/CineMA/cinema/regression/train.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cardio.data.transform import get_patch_grid, patch_grid_sample
from cardio.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


# =========================================================================
# Loss
# =========================================================================


def regression_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE loss for regression tasks.

    Args:
        model: regression model.
        batch: batch dict.
        views: list of view names.
        device: target device.

    Returns:
        Tuple of ``(loss, metrics_dict)``.
    """
    image_dict = {v: batch[f"{v}_image"].to(device) for v in views}
    preds = model(image_dict)
    label = batch["label"].to(dtype=preds.dtype, device=device)
    mse = F.mse_loss(preds, label)
    mae = F.l1_loss(preds, label)
    metrics = {
        "mse_loss": mse.item(),
        "loss": mse.item(),
        "mae_loss": mae.item(),
        "max_label": torch.max(label).item(),
        "min_label": torch.min(label).item(),
        "max_pred": torch.max(preds).item(),
        "min_pred": torch.min(preds).item(),
    }
    return mse, metrics


# =========================================================================
# Inference / evaluation
# =========================================================================


def regression_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Forward pass with optional patch-based inference.

    Args:
        model: regression model.
        image_dict: ``{view: (1, C, *spatial)}``.
        patch_size_dict: expected patch size per view.
        amp_dtype: AMP dtype.

    Returns:
        Predictions tensor ``(1, 1)``.
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

    patch_preds_list: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {v: patch if v == view_to_patch else image_dict[v] for v in views}
            patch_preds_list.append(model(patch_image_dict))

    return torch.mean(torch.cat(patch_preds_list), dim=0, keepdim=True)


# =========================================================================
# Metrics
# =========================================================================


def regression_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, etc.).

    Args:
        true_labels: ``(n_samples,)``.
        pred_labels: ``(n_samples,)``.

    Returns:
        Metrics dict.
    """
    abs_error = np.abs(true_labels - pred_labels)
    return {
        "rmse": float(np.sqrt(np.mean(abs_error**2))),
        "mae": float(np.mean(abs_error)),
        "max_error": float(np.max(abs_error)),
        "min_error": float(np.min(abs_error)),
        "max_label": float(np.max(true_labels)),
        "min_label": float(np.min(true_labels)),
        "max_pred": float(np.max(pred_labels)),
        "min_pred": float(np.min(pred_labels)),
    }


def regression_eval(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate regression model on one batch.

    Args:
        model: regression model.
        batch: data batch.
        patch_size_dict: patch sizes per view.
        amp_dtype: AMP dtype.
        device: target device.

    Returns:
        Tuple of ``(predictions, empty_metrics)``.
    """
    views = list(patch_size_dict.keys())
    image_dict = {v: batch[f"{v}_image"].to(device) for v in views}
    preds = regression_forward(model, image_dict, patch_size_dict, amp_dtype)
    return preds, {}


def regression_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],  # noqa: ARG001
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate regression model over a full dataloader.

    Args:
        model: regression model.
        dataloader: validation data loader.
        patch_size_dict: patch sizes per view.
        spacing_dict: (unused, kept for API compatibility with ``run_train``).
        amp_dtype: AMP dtype.
        device: target device.

    Returns:
        Aggregated regression metrics (normalised and restored).
    """
    pred_labels_list: list[torch.Tensor] = []
    true_labels_list: list[torch.Tensor] = []

    for _, batch in enumerate(dataloader):
        preds, _ = regression_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device,
        )
        pred_labels_list.append(preds)
        true_labels_list.append(batch["label"])

    pred_labels = torch.cat(pred_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()

    reg_mean = dataloader.dataset.reg_mean
    reg_std = dataloader.dataset.reg_std
    restored_pred = (pred_labels * reg_std) + reg_mean
    restored_true = (true_labels * reg_std) + reg_mean

    metrics = regression_metrics(true_labels=true_labels, pred_labels=pred_labels)
    restored = regression_metrics(true_labels=restored_true, pred_labels=restored_pred)
    metrics.update({f"restored_{k}": v for k, v in restored.items()})
    return metrics
