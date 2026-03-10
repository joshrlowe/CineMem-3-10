"""Evaluation utilities for CardioVLM.

Re-exports from sub-modules:

* :mod:`cardio.eval.metric` -- shared metric helpers
* :mod:`cardio.eval.segmentation` -- segmentation evaluation
* :mod:`cardio.eval.vlm` -- basic VLM evaluation
* :mod:`cardio.eval.vlm_cardiac` -- cardiac-specific VLM evaluation
"""

from cardio.eval.metric import aggregate_scores, compute_ef_metrics
from cardio.eval.segmentation import (
    get_ejection_fraction,
    process_ef_metrics,
    process_mean_metrics,
    save_segmentation_metrics,
    segmentation_metrics,
)
from cardio.eval.vlm import evaluate_vlm
from cardio.eval.vlm_cardiac import evaluate_vlm_cardiac

__all__ = [
    "aggregate_scores",
    "compute_ef_metrics",
    "evaluate_vlm",
    "evaluate_vlm_cardiac",
    "get_ejection_fraction",
    "process_ef_metrics",
    "process_mean_metrics",
    "save_segmentation_metrics",
    "segmentation_metrics",
]
