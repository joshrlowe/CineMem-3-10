"""Segmentation evaluation metrics and reporting.

Ported from ``_reference/CineMA/cinema/segmentation/eval.py`` and
``_reference/CineMA/cinema/segmentation/train.py``.  Only metric
computation and CSV reporting are included; the full model-loading /
dataset pipeline is part of the vision training codebase and should use
:class:`~cardio.tools.segment.CardiacSegmentationTool` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from sklearn.metrics import accuracy_score, matthews_corrcoef, root_mean_squared_error
from torch.nn import functional as F  # noqa: N812

from cardio.data.constants import LV_LABEL
from cardio.utils.logging import get_logger
from cardio.vision.metric import ejection_fraction, get_ef_region, get_volumes, stability_score

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


# =========================================================================
# Per-batch segmentation metrics
# =========================================================================


def segmentation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Compute evaluation metrics for a segmentation batch.

    Padded regions are treated as background.  The background class is at
    index 0 in the one-hot encoding; MONAI only supports up to 3-D.

    Args:
        logits: raw predictions ``(batch, 1+n_classes, ...)``.
        labels: ground-truth labels ``(batch, 1, ...)``.
        spacing: pixel / voxel spacing in mm.

    Returns:
        Dict where each value has shape ``(batch,)``.
    """
    n_classes = logits.shape[1] - 1
    labels = labels.squeeze(dim=1).long()

    pred_labels = torch.argmax(logits, dim=1)
    pred_mask = F.one_hot(pred_labels, n_classes + 1).moveaxis(-1, 1)
    true_mask = F.one_hot(labels, n_classes + 1).moveaxis(-1, 1)

    dice = compute_dice(y_pred=pred_mask, y=true_mask, num_classes=n_classes + 1)
    iou = compute_iou(y_pred=pred_mask, y=true_mask)
    stability = stability_score(logits=logits)
    hausdorff_dist = compute_hausdorff_distance(
        y_pred=pred_mask, y=true_mask, percentile=95, spacing=spacing,
    )
    true_volumes = get_volumes(mask=true_mask, spacing=spacing)
    pred_volumes = get_volumes(mask=pred_mask, spacing=spacing)

    metrics: dict[str, torch.Tensor] = {}
    for i in range(n_classes):
        cls_idx = i + 1
        metrics[f"class_{cls_idx}_dice_score"] = dice[:, cls_idx]
        metrics[f"class_{cls_idx}_iou_score"] = iou[:, cls_idx]
        metrics[f"class_{cls_idx}_stability_score"] = stability[:, cls_idx]
        metrics[f"class_{cls_idx}_hausdorff_distance_95"] = hausdorff_dist[:, cls_idx - 1]
        metrics[f"class_{cls_idx}_true_volume"] = true_volumes[:, cls_idx]
        metrics[f"class_{cls_idx}_pred_volume"] = pred_volumes[:, cls_idx]

    metrics["mean_dice_score"] = torch.mean(dice[:, 1:], dim=-1)
    metrics["mean_iou_score"] = torch.mean(iou[:, 1:], dim=-1)
    metrics["mean_stability_score"] = torch.mean(stability[:, 1:], dim=-1)
    metrics["mean_hausdorff_distance_95"] = torch.mean(hausdorff_dist, dim=-1)

    return metrics


# =========================================================================
# Ejection-fraction derivation from metric DataFrames
# =========================================================================


def get_ejection_fraction(
    metric_df: pd.DataFrame,
    views: list[str],
) -> pd.DataFrame:
    """Merge ED/ES metric rows to compute true and predicted EF per patient.

    Args:
        metric_df: DataFrame where each row is one ED or ES slice.
            Must contain columns ``pid``, ``is_ed``, and per-view volume
            columns following the naming convention from
            :func:`segmentation_metrics`.
        views: list of view identifiers (e.g. ``["sax"]``).

    Returns:
        DataFrame with one row per patient.
    """
    columns = ["pid", f"class_{LV_LABEL}_true_volume", f"class_{LV_LABEL}_pred_volume"]
    for view in views:
        columns += [f"{view}_class_{LV_LABEL}_true_volume", f"{view}_class_{LV_LABEL}_pred_volume"]

    ed_df = metric_df[metric_df["is_ed"]][columns].set_index("pid").add_prefix("ed_")
    es_df = metric_df[~metric_df["is_ed"]][columns].set_index("pid").add_prefix("es_")
    ef_df = ed_df.merge(es_df, on="pid")

    prefixes = [f"{view}_" for view in views]
    for p in ["", *prefixes]:
        ef_df = ef_df.rename(
            columns={
                f"ed_{p}class_{LV_LABEL}_true_volume": f"{p}true_edv",
                f"ed_{p}class_{LV_LABEL}_pred_volume": f"{p}pred_edv",
                f"es_{p}class_{LV_LABEL}_true_volume": f"{p}true_esv",
                f"es_{p}class_{LV_LABEL}_pred_volume": f"{p}pred_esv",
            },
            errors="raise",
        )
        ef_df[f"{p}true_ef"] = ejection_fraction(
            edv=ef_df[f"{p}true_edv"].to_numpy(),
            esv=ef_df[f"{p}true_esv"].to_numpy(),
        )
        ef_df[f"{p}pred_ef"] = ejection_fraction(
            edv=ef_df[f"{p}pred_edv"].to_numpy(),
            esv=ef_df[f"{p}pred_esv"].to_numpy(),
        )
        ef_df[f"{p}pred_ef"] = (
            ef_df[f"{p}pred_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
        )
        ef_df[f"{p}true_ef"] = (
            ef_df[f"{p}true_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
        )
        ef_df[f"{p}ef_error"] = abs(ef_df[f"{p}pred_ef"] - ef_df[f"{p}true_ef"])

    return ef_df


# =========================================================================
# Metric reporting / CSV export
# =========================================================================


def process_mean_metrics(
    metric_df: pd.DataFrame,
    metric_path: Path,
) -> None:
    """Aggregate mean and std of all numeric metrics and save to CSV.

    Args:
        metric_df: DataFrame with columns ``pid``, ``is_ed``, and
            assorted metric columns.
        metric_path: destination CSV path.
    """
    numeric = metric_df.drop(columns=["pid", "is_ed"])
    mean_metrics = {f"{k}_mean": v for k, v in numeric.mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in numeric.std().to_dict().items()}
    pd.DataFrame([{**mean_metrics, **std_metrics}]).T.to_csv(metric_path, header=False)
    logger.info("Saved mean metrics to %s.", metric_path)


def process_ef_metrics(
    ef_df: pd.DataFrame,
    views: list[str],
    metric_path: Path,
) -> None:
    """Compute ejection-fraction summary metrics and save to CSV.

    Computes MAE, RMSE, accuracy (region-binned), and MCC for each
    view prefix as well as the overall (unprefixed) EF.

    Args:
        ef_df: per-patient DataFrame from :func:`get_ejection_fraction`.
        views: list of view identifiers.
        metric_path: destination CSV path.
    """
    prefixes = [f"{view}_" for view in views]
    metrics: dict[str, float] = {}
    for prefix in ["", *prefixes]:
        ef_true_labels = ef_df[f"{prefix}true_ef"].apply(get_ef_region)
        ef_pred_labels = ef_df[f"{prefix}pred_ef"].apply(get_ef_region)
        ef_metrics = {
            f"{prefix}ef_mae": ef_df[f"{prefix}ef_error"].mean(),
            f"{prefix}edv_mae": (ef_df[f"{prefix}true_edv"] - ef_df[f"{prefix}pred_edv"]).abs().mean(),
            f"{prefix}esv_mae": (ef_df[f"{prefix}true_esv"] - ef_df[f"{prefix}pred_esv"]).abs().mean(),
            f"{prefix}ef_err_std": ef_df[f"{prefix}ef_error"].std(),
            f"{prefix}edv_err_std": (ef_df[f"{prefix}true_edv"] - ef_df[f"{prefix}pred_edv"]).abs().std(),
            f"{prefix}esv_err_std": (ef_df[f"{prefix}true_esv"] - ef_df[f"{prefix}pred_esv"]).abs().std(),
            f"{prefix}ef_rmse": root_mean_squared_error(ef_df[f"{prefix}true_ef"], ef_df[f"{prefix}pred_ef"]),
            f"{prefix}edv_rmse": root_mean_squared_error(ef_df[f"{prefix}true_edv"], ef_df[f"{prefix}pred_edv"]),
            f"{prefix}esv_rmse": root_mean_squared_error(ef_df[f"{prefix}true_esv"], ef_df[f"{prefix}pred_esv"]),
            f"{prefix}ef_acc": accuracy_score(y_true=ef_true_labels, y_pred=ef_pred_labels),
            f"{prefix}ef_mcc": matthews_corrcoef(y_true=ef_true_labels, y_pred=ef_pred_labels),
        }
        metrics.update(ef_metrics)
    pd.DataFrame([metrics]).T.to_csv(metric_path, header=False)
    logger.info("Saved EF metrics to %s.", metric_path)


def save_segmentation_metrics(
    metric_df: pd.DataFrame,
    views: list[str],
    out_dir: Path,
) -> None:
    """Save all segmentation metrics to *out_dir*.

    Writes three CSV files:

    * ``metrics.csv`` -- raw per-slice metrics.
    * ``mean_metrics.csv`` -- aggregated mean/std.
    * ``mean_ef_metrics.csv`` -- ejection-fraction summary.

    Args:
        metric_df: per-slice DataFrame.
        views: list of view identifiers.
        out_dir: output directory (created if necessary).
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    metric_path = out_dir / "metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    logger.info("Saved metrics to %s.", metric_path)

    logger.info("Mean metrics across ED and ES slices.")
    process_mean_metrics(metric_df, out_dir / "mean_metrics.csv")

    metric_path = out_dir / "ef_metrics.csv"
    ef_df = get_ejection_fraction(metric_df, views=views)
    ef_df.to_csv(metric_path, index=True)
    logger.info("Saved ejection fraction metrics to %s.", metric_path)

    logger.info("Ejection fraction metrics across patients.")
    process_ef_metrics(ef_df, views=views, metric_path=out_dir / "mean_ef_metrics.csv")
