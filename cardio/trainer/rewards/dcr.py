"""Divide-Conquer-Reasoning evaluator for visual grounding verification.

Decomposes a VLM reasoning trace into atomic anatomical claims, then
verifies each claim's spatial grounding by computing bounding-box IoU
against ground-truth segmentation masks.

The companion :class:`AutoMetricConverter` maps raw IoU scores to
continuous penalty values and produces per-token penalty masks consumed
by TR-GRPO for localised advantage assignment.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

from cardio.data.io.sitk import get_binary_mask_bounding_box

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Anatomy claim patterns
# ---------------------------------------------------------------------------

ANATOMY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+"
        r"(?:wall|segment|region)\s+"
        r"(?:demonstrates?|shows?|exhibits?|has)\s+"
        r"(.+?)(?:\.|,|;)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the\s+)?(LV|RV|LA|RA|septum|apex|base)\s+"
        r"(?:is|appears?)\s+"
        r"(.+?)(?:\.|,|;)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:focal|regional|global)\s+(\w+)\s+"
        r"(?:in|of|at)\s+(?:the\s+)?"
        r"(.+?)(?:\.|,|;)",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Anatomy alias normalisation
# ---------------------------------------------------------------------------

_ANATOMY_ALIASES: dict[str, str] = {
    "left ventricle": "LV",
    "left ventricular": "LV",
    "lv": "LV",
    "lv cavity": "LV",
    "right ventricle": "RV",
    "right ventricular": "RV",
    "rv": "RV",
    "rv cavity": "RV",
    "myocardium": "MYO",
    "myocardial": "MYO",
    "myo": "MYO",
    "left atrium": "LA",
    "left atrial": "LA",
    "la": "LA",
    "right atrium": "RA",
    "right atrial": "RA",
    "ra": "RA",
    "septum": "septum",
    "interventricular septum": "septum",
    "septal": "septum",
    "apex": "apex",
    "apical": "apex",
    "base": "base",
    "basal": "base",
    "anterior": "anterior",
    "anterior wall": "anterior",
    "inferior": "inferior",
    "inferior wall": "inferior",
    "lateral": "lateral",
    "lateral wall": "lateral",
}


def _normalise_anatomy(raw: str) -> str:
    """Map a free-text anatomy reference to its canonical key."""
    key = raw.strip().lower()
    return _ANATOMY_ALIASES.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Bbox IoU helpers
# ---------------------------------------------------------------------------


def _mask_to_bbox_xyxy(mask: np.ndarray) -> list[int]:
    """Convert a 2-D binary mask to ``[x1, y1, x2, y2]``.

    Uses the project-standard row/col-to-x/y swap (x = col, y = row).
    Returns ``[-1, -1, -1, -1]`` when the mask has no foreground.
    """
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask)
    if int(bbox_min[0]) == -1:
        return [-1, -1, -1, -1]
    return [int(bbox_min[1]), int(bbox_min[0]), int(bbox_max[1]), int(bbox_max[0])]


def _bbox_iou(box_a: list[float], box_b: list[int]) -> float:
    """Axis-aligned IoU between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return float(inter_area / union)


# =========================================================================
# DivideConquerEvaluator
# =========================================================================


class DivideConquerEvaluator:
    """Decompose a reasoning trace into atomic anatomical claims and
    evaluate each claim's visual grounding via bounding-box IoU."""

    # ------------------------------------------------------------------
    # Claim isolation
    # ------------------------------------------------------------------

    def isolate_claims(self, text: str) -> list[dict]:
        """Break reasoning text into atomic sentence-level claims.

        Returns a list of dicts with keys ``sentence``, ``anatomy``,
        ``finding``, ``start_char``, ``end_char``.
        """
        claims: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for pattern in ANATOMY_PATTERNS:
            for m in pattern.finditer(text):
                anatomy_raw = m.group(1)
                finding = m.group(2).strip()
                start_char = m.start()
                end_char = m.end()

                anatomy = _normalise_anatomy(anatomy_raw)
                dedup_key = (anatomy, start_char)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                claims.append({
                    "sentence": m.group(0).strip(),
                    "anatomy": anatomy,
                    "finding": finding,
                    "start_char": start_char,
                    "end_char": end_char,
                })

        claims.sort(key=lambda c: c["start_char"])
        return claims

    # ------------------------------------------------------------------
    # IoU computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iou(
        predicted_bbox: list[float],
        ground_truth_mask: np.ndarray,
    ) -> float:
        """IoU between a predicted bbox and the bbox of a GT mask.

        Args:
            predicted_bbox: ``[x1, y1, x2, y2]`` predicted bounding box.
            ground_truth_mask: 2-D numpy array (binary/boolean).

        Returns:
            IoU score in ``[0, 1]``.
        """
        gt_bbox = _mask_to_bbox_xyxy(ground_truth_mask)
        if gt_bbox == [-1, -1, -1, -1]:
            return 0.0
        return _bbox_iou(predicted_bbox, gt_bbox)

    # ------------------------------------------------------------------
    # Claim evaluation
    # ------------------------------------------------------------------

    def evaluate_claims(
        self,
        claims: list[dict],
        predicted_bboxes: dict[str, list[float]],
        ground_truth_masks: dict[str, np.ndarray],
    ) -> list[dict]:
        """Annotate each claim with its IoU score.

        Args:
            claims: output of :meth:`isolate_claims`.
            predicted_bboxes: mapping from anatomy key to predicted
                ``[x1, y1, x2, y2]``.
            ground_truth_masks: mapping from anatomy key to 2-D binary
                numpy mask.

        Returns:
            A copy of *claims* with added ``iou`` and
            ``grounding_status`` fields.
        """
        results: list[dict] = []
        for claim in claims:
            entry = dict(claim)
            anatomy = claim["anatomy"]

            pred_box = predicted_bboxes.get(anatomy)
            gt_mask = ground_truth_masks.get(anatomy)

            if pred_box is None or gt_mask is None:
                entry["iou"] = 0.0
                entry["grounding_status"] = "missing"
                results.append(entry)
                continue

            iou = self.compute_iou(pred_box, gt_mask)
            entry["iou"] = iou

            if iou >= 0.5:
                entry["grounding_status"] = "correct"
            elif iou >= 0.2:
                entry["grounding_status"] = "partial"
            else:
                entry["grounding_status"] = "hallucination"

            results.append(entry)

        return results


# =========================================================================
# AutoMetricConverter
# =========================================================================


class AutoMetricConverter:
    """Convert raw IoU scores into continuous penalty values for TR-GRPO.

    Mapping:
        - ``IoU >= 0.5``  ->  ``0.0`` (correct grounding)
        - ``0.2 <= IoU < 0.5``  ->  ``1.0 - (IoU / 0.5)`` (partial)
        - ``IoU < 0.2``  ->  ``1.0`` (hallucination)
    """

    @staticmethod
    def compute_penalty(iou: float) -> float:
        """Map a single IoU score to a penalty in ``[0, 1]``."""
        if iou >= 0.5:
            return 0.0
        if iou >= 0.2:
            return 1.0 - (iou / 0.5)
        return 1.0

    @staticmethod
    def build_token_mask(
        claims_with_penalties: list[dict],
        token_offsets: list[tuple[int, int]],
        seq_len: int,
    ) -> torch.Tensor:
        """Build a per-token penalty mask of shape ``(seq_len,)``.

        Tokens whose character span overlaps a penalised claim receive
        that claim's penalty value.  When multiple claims overlap, the
        maximum penalty is used.

        Args:
            claims_with_penalties: list of claim dicts, each containing
                at least ``start_char``, ``end_char``, and ``penalty``.
            token_offsets: list of ``(start_char, end_char)`` tuples, one
                per token in the sequence.
            seq_len: total sequence length (may exceed
                ``len(token_offsets)`` due to special tokens).

        Returns:
            ``torch.Tensor`` of shape ``(seq_len,)`` with dtype
            ``torch.float32``.
        """
        mask = torch.zeros(seq_len, dtype=torch.float32)

        for claim in claims_with_penalties:
            penalty = claim.get("penalty", 0.0)
            if penalty <= 0.0:
                continue

            c_start = claim["start_char"]
            c_end = claim["end_char"]

            for tok_idx, (t_start, t_end) in enumerate(token_offsets):
                if t_start < c_end and t_end > c_start:
                    mask[tok_idx] = max(mask[tok_idx].item(), penalty)

        return mask
