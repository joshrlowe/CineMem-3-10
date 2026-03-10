"""Evaluation metrics for CardioVLM.

Re-exports core cardiac metrics from :mod:`cardio.vision.metric` and adds
VLM-specific evaluation helpers (EF accuracy, score aggregation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, root_mean_squared_error

from cardio.vision.metric import (
    coefficient_of_variance,
    ejection_fraction,
    get_ef_region,
    get_volumes,
)

__all__ = [
    "aggregate_scores",
    "coefficient_of_variance",
    "compute_ef_metrics",
    "ejection_fraction",
    "get_ef_region",
    "get_volumes",
]


def compute_ef_metrics(
    true_ef: np.ndarray,
    pred_ef: np.ndarray,
) -> dict[str, float]:
    """Compute ejection-fraction evaluation metrics.

    Args:
        true_ef: ground-truth EF values (%).
        pred_ef: predicted EF values (%).

    Returns:
        Dict with ``ef_mae``, ``ef_rmse``, ``ef_acc`` (region-binned),
        and ``ef_mcc``.
    """
    true_ef = np.asarray(true_ef, dtype=np.float64)
    pred_ef = np.asarray(pred_ef, dtype=np.float64)

    ef_error = np.abs(true_ef - pred_ef)
    true_labels = [get_ef_region(float(v)) for v in true_ef]
    pred_labels = [get_ef_region(float(v)) for v in pred_ef]

    return {
        "ef_mae": float(np.mean(ef_error)),
        "ef_rmse": float(root_mean_squared_error(true_ef, pred_ef)),
        "ef_err_std": float(np.std(ef_error)),
        "ef_acc": float(accuracy_score(y_true=true_labels, y_pred=pred_labels)),
        "ef_mcc": float(matthews_corrcoef(y_true=true_labels, y_pred=pred_labels)),
    }


def aggregate_scores(
    scores_by_category: dict[str, list[float]],
) -> dict[str, Any]:
    """Aggregate per-category score lists into summary statistics.

    Args:
        scores_by_category: mapping from category name to list of
            per-sample scores.

    Returns:
        Dict with ``overall`` (mean/std/n across all samples) and
        ``per_category`` (mean/std/n per category).
    """
    all_scores: list[float] = []
    per_cat: dict[str, dict[str, float | int]] = {}

    for cat in sorted(scores_by_category):
        vals = scores_by_category[cat]
        arr = np.array(vals, dtype=np.float64)
        per_cat[cat] = {
            "mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
            "std": float(np.std(arr)) if len(arr) > 0 else 0.0,
            "n": len(vals),
        }
        all_scores.extend(vals)

    all_arr = np.array(all_scores, dtype=np.float64)
    overall = {
        "mean": float(np.mean(all_arr)) if len(all_arr) > 0 else 0.0,
        "std": float(np.std(all_arr)) if len(all_arr) > 0 else 0.0,
        "n": len(all_scores),
    }

    return {"overall": overall, "per_category": per_cat}
