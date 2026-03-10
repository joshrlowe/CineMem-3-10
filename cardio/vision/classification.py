"""Classification model training and evaluation utilities.

Ported from ``_reference/CineMA/cinema/classification/train.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from torch.nn import functional as F  # noqa: N812

from cardio.data.transform import get_patch_grid, patch_grid_sample
from cardio.utils.logging import get_logger
from cardio.vision.convvit import ConvViT
from cardio.vision.resnet import get_resnet2d, get_resnet3d
from cardio.vision.vit import get_vit_config

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


# =========================================================================
# Model factory
# =========================================================================


def get_classification_or_regression_model(config: DictConfig) -> nn.Module:
    """Return the classification or regression model specified by *config*.

    The number of output channels is inferred from ``config.data.class_column``
    (classification) or ``config.data.regression_column`` (regression),
    falling back to ``config.model.out_chans``.

    Args:
        config: OmegaConf configuration.

    Returns:
        Instantiated model.
    """
    if hasattr(config.data, "class_column"):
        out_chans = len(config.data[config.data.class_column])
    elif hasattr(config.data, "regression_column"):
        out_chans = 1
    else:
        logger.info("Using config.model.out_chans %d.", config.model.out_chans)
        out_chans = config.model.out_chans
    views = [config.model.views] if isinstance(config.model.views, str) else list(config.model.views)
    in_chans_dict = {v: config.data.sax.in_chans if v == "sax" else config.data.lax.in_chans for v in views}

    if config.model.name == "convvit":
        vit_config = get_vit_config(config.model.convvit.size)
        image_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
        ndim_dict = {v: 3 if v == "sax" else 2 for v in views}
        enc_patch_size_dict = {v: config.model.convvit.enc_patch_size[:n] for v, n in ndim_dict.items()}
        enc_scale_factor_dict = {v: config.model.convvit.enc_scale_factor[:n] for v, n in ndim_dict.items()}
        model = ConvViT(
            image_size_dict=image_size_dict,
            n_frames=config.model.n_frames,
            in_chans_dict=in_chans_dict,
            out_chans=out_chans,
            enc_patch_size_dict=enc_patch_size_dict,
            enc_scale_factor_dict=enc_scale_factor_dict,
            enc_conv_chans=config.model.convvit.enc_conv_chans,
            enc_conv_n_blocks=config.model.convvit.enc_conv_n_blocks,
            enc_embed_dim=vit_config["enc_embed_dim"],
            enc_depth=vit_config["enc_depth"],
            enc_n_heads=vit_config["enc_n_heads"],
            drop_path=config.model.convvit.drop_path,
        )
    elif config.model.name == "resnet":
        if len(views) > 1:
            msg = "ResNet only supports single view."
            raise ValueError(msg)
        view = views[0]
        get_fn = get_resnet3d if view == "sax" else get_resnet2d
        model = get_fn(
            depth=config.model.resnet.depth,
            in_chans=in_chans_dict[view] * config.model.n_frames,
            out_chans=out_chans,
            layer_inplanes=config.model.resnet.layer_inplanes,
        )
    else:
        msg = f"Invalid model name {config.model.name}."
        raise ValueError(msg)

    if hasattr(model, "set_grad_ckpt"):
        model.set_grad_ckpt(config.grad_ckpt)
    return model


# =========================================================================
# Loss
# =========================================================================


def classification_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy loss with label smoothing.

    Args:
        model: classification model.
        batch: batch dict.
        views: list of view names.
        device: target device.
        label_smoothing: smoothing factor.

    Returns:
        Tuple of ``(loss, metrics_dict)``.
    """
    image_dict = {v: batch[f"{v}_image"].to(device) for v in views}
    logits = model(image_dict)
    label = batch["label"].long().to(device)
    ce = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
    metrics = {"cross_entropy": ce.item(), "loss": ce.item()}
    return ce, metrics


# =========================================================================
# Inference / evaluation
# =========================================================================


def classification_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Forward pass with optional patch-based inference.

    Args:
        model: classification model.
        image_dict: ``{view: (1, C, *spatial)}``.
        patch_size_dict: expected patch size per view.
        amp_dtype: AMP dtype.

    Returns:
        Logits tensor ``(1, n_classes)``.
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

    patch_probs_list: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {v: patch if v == view_to_patch else image_dict[v] for v in views}
            patch_logits = model(patch_image_dict)
            patch_probs_list.append(F.softmax(patch_logits, dim=1))
    patch_probs = torch.cat(patch_probs_list, dim=0)
    return torch.log(torch.mean(patch_probs, dim=0, keepdim=True))


# =========================================================================
# Metrics
# =========================================================================


def binary_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        true_labels: ``(n_samples,)``.
        pred_labels: ``(n_samples,)``.
        pred_probs: ``(n_samples, 2)``.
        n_classes: must be 2.

    Returns:
        Metrics dict.
    """
    if n_classes != 2:
        msg = f"Expected n_classes=2, got {n_classes}."
        raise ValueError(msg)
    labels = list(range(n_classes))
    metrics: dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    metrics["entropy"] = -np.mean(np.sum(pred_probs * np.log(pred_probs + 1e-6), axis=1))
    cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp)
    metrics["sensitivity"] = tp / (tp + fn)
    metrics["f1"] = f1_score(y_true=true_labels, y_pred=pred_labels, labels=labels)
    if len(np.unique(true_labels)) > 1:
        metrics["mcc"] = matthews_corrcoef(y_true=true_labels, y_pred=pred_labels)
        metrics["roc_auc"] = roc_auc_score(y_true=true_labels, y_score=pred_probs[:, 1], labels=labels)
    else:
        metrics["mcc"] = 0.0
        metrics["roc_auc"] = 0.0
    return metrics


def multiclass_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Compute multiclass classification metrics.

    Args:
        true_labels: ``(n_samples,)``.
        pred_labels: ``(n_samples,)``.
        pred_probs: ``(n_samples, n_classes)``.
        n_classes: number of classes.

    Returns:
        Metrics dict.
    """
    labels = list(range(n_classes))
    metrics: dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    metrics["entropy"] = -np.mean(np.sum(pred_probs * np.log(pred_probs + 1e-6), axis=1))
    metrics["f1"] = f1_score(y_true=true_labels, y_pred=pred_labels, average="micro", labels=labels)
    if len(np.unique(true_labels)) > 1:
        metrics["mcc"] = matthews_corrcoef(y_true=true_labels, y_pred=pred_labels)
        metrics["roc_auc"] = roc_auc_score(
            y_true=true_labels, y_score=pred_probs, average="macro", multi_class="ovo", labels=labels,
        )
    else:
        metrics["mcc"] = 0.0
        metrics["roc_auc"] = 0.0
    return metrics


def classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> dict[str, float]:
    """Dispatch to binary or multiclass metrics.

    Args:
        true_labels: ``(n_samples,)``.
        pred_labels: ``(n_samples,)``.
        pred_probs: ``(n_samples, n_classes)``.

    Returns:
        Metrics dict.
    """
    n_classes = pred_probs.shape[1]
    if n_classes == 2:
        return binary_classification_metrics(true_labels, pred_labels, pred_probs, n_classes)
    return multiclass_classification_metrics(true_labels, pred_labels, pred_probs, n_classes)


def classification_eval(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate classification model on one batch.

    Args:
        model: classification model.
        batch: data batch.
        patch_size_dict: patch sizes per view.
        amp_dtype: AMP dtype.
        device: target device.

    Returns:
        Tuple of ``(logits, empty_metrics)``.
    """
    views = list(patch_size_dict.keys())
    image_dict = {v: batch[f"{v}_image"].to(device) for v in views}
    logits = classification_forward(model, image_dict, patch_size_dict, amp_dtype)
    return logits, {}


def classification_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],  # noqa: ARG001
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classification model over a full dataloader.

    Args:
        model: classification model.
        dataloader: validation data loader.
        patch_size_dict: patch sizes per view.
        spacing_dict: (unused, kept for API compatibility with ``run_train``).
        amp_dtype: AMP dtype.
        device: target device.

    Returns:
        Aggregated classification metrics.
    """
    pred_labels_list: list[torch.Tensor] = []
    true_labels_list: list[torch.Tensor] = []
    pred_logits_list: list[torch.Tensor] = []

    for _, batch in enumerate(dataloader):
        logits, _ = classification_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device,
        )
        pred_labels_list.append(torch.argmax(logits, dim=1))
        true_labels_list.append(batch["label"])
        pred_logits_list.append(logits)

    pred_labels = torch.cat(pred_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()
    pred_logits = torch.cat(pred_logits_list, dim=0).cpu().to(dtype=torch.float32)
    pred_probs = F.softmax(pred_logits, dim=1).numpy()
    return classification_metrics(true_labels=true_labels, pred_labels=pred_labels, pred_probs=pred_probs)
