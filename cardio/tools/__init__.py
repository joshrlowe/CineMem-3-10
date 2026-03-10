"""Cardiac tool backends and dispatcher.

Provides :class:`ToolRouter` which parses ``<tool_call>`` JSON from
VLM-generated text and dispatches to the appropriate backend:

- :class:`CardiacSegmentationTool` -- runs ConvUNetR segmentation
- :class:`CardiacMeasurementTool` -- computes clinical measurements
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import torch

from cardio.data.constants import LV_LABEL
from cardio.tools.measure import CardiacMeasurementTool
from cardio.tools.segment import CardiacSegmentationTool
from cardio.utils.logging import get_logger
from cardio.vision.metric import get_volumes

logger = get_logger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL,
)


class ToolRouter:
    """Parse ``<tool_call>`` blocks and dispatch to tool backends.

    Tool instances can be provided at construction time or will be
    created lazily on first use.  Lazy construction avoids loading
    heavy models (ConvUNetR weights) until a tool call is actually
    encountered.

    Args:
        segmentation_tool: pre-initialised
            :class:`CardiacSegmentationTool`, or ``None`` for lazy init.
        measurement_tool: pre-initialised
            :class:`CardiacMeasurementTool`, or ``None`` for lazy init.
    """

    def __init__(
        self,
        segmentation_tool: CardiacSegmentationTool | None = None,
        measurement_tool: CardiacMeasurementTool | None = None,
    ) -> None:
        """Initialise the router with optional pre-built backends."""
        self._seg_tool = segmentation_tool
        self._meas_tool = measurement_tool

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def segmentation_tool(self) -> CardiacSegmentationTool:
        """Return (and lazily create) the segmentation backend."""
        if self._seg_tool is None:
            self._seg_tool = CardiacSegmentationTool()
        return self._seg_tool

    @property
    def measurement_tool(self) -> CardiacMeasurementTool:
        """Return (and lazily create) the measurement backend."""
        if self._meas_tool is None:
            self._meas_tool = CardiacMeasurementTool()
        return self._meas_tool

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract all ``<tool_call>`` blocks from *text*.

        Each block is expected to contain JSON of the form
        ``{"name": "<tool_name>", "args": {...}}``.  Malformed blocks
        are logged and skipped.

        Returns:
            List of parsed tool-call dicts.
        """
        results: list[dict[str, Any]] = []
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed <tool_call>: {raw!r}")
                continue
            if not isinstance(parsed, dict) or "name" not in parsed:
                logger.warning(f"Skipping <tool_call> without 'name': {raw!r}")
                continue
            parsed.setdefault("args", {})
            results.append(parsed)
        return results

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_call: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch a single parsed tool call to its backend.

        Args:
            tool_call: parsed dict with ``name`` and ``args`` keys.
            context: runtime context, e.g. ``{"nifti_path": "...",
                "ed_mask": ..., "es_mask": ..., "spacing": ...}``.

        Returns:
            Result dict from the backend, suitable for formatting as
            ``<tool_result>``.

        Raises:
            ValueError: if the tool name is not recognised.
        """
        name = tool_call["name"]
        args = tool_call.get("args", {})

        if name == "segment_cardiac":
            return self._exec_segment(args, context)
        if name == "measure_volume":
            return self._exec_measure(args, context)

        msg = f"Unknown tool: {name!r}"
        raise ValueError(msg)

    def execute_all(
        self,
        text: str,
        context: dict[str, Any],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Parse and execute every ``<tool_call>`` in *text*.

        Returns:
            List of ``(tool_call_dict, result_dict)`` pairs.
        """
        calls = self.parse_tool_calls(text)
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for tc in calls:
            try:
                result = self.execute(tc, context)
            except Exception:
                logger.exception(f"Tool execution failed for {tc!r}")
                result = {"status": "error", "tool": tc.get("name", "unknown")}
            results.append((tc, result))
        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_result(result: dict[str, Any]) -> str:
        """Wrap a result dict as ``<tool_result>JSON</tool_result>``."""
        return f"<tool_result>{json.dumps(result)}</tool_result>"

    # ------------------------------------------------------------------
    # Internal dispatch helpers
    # ------------------------------------------------------------------

    def _exec_segment(
        self,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle ``segment_cardiac`` tool calls."""
        nifti_path = context.get("nifti_path")
        if nifti_path is None:
            return {"status": "error", "message": "nifti_path not in context"}

        frame_idx = args.get("frame_idx")
        result = self.segmentation_tool.segment(nifti_path, frame_idx=frame_idx)

        # Strip the heavy mask array for the JSON-serialisable result;
        # store it in context for downstream measurement calls.
        mask = result.pop("mask")
        spacing = result.pop("spacing")
        context["last_seg_mask"] = mask
        context["spacing"] = spacing

        structures = args.get("structures", ["LV", "RV", "MYO"])
        return {
            "structures": structures,
            "status": "success",
            "lv_area": result.get("lv_area", 0),
            "rv_area": result.get("rv_area", 0),
            "myo_area": result.get("myo_area", 0),
            "bbox_lv": result.get("bbox_lv"),
            "bbox_rv": result.get("bbox_rv"),
        }

    def _exec_measure(
        self,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle ``measure_volume`` tool calls."""
        ed_mask = context.get("ed_mask") or context.get("last_seg_mask")
        es_mask = context.get("es_mask")
        spacing = context.get("spacing")

        if ed_mask is None or spacing is None:
            return {
                "status": "error",
                "message": "Missing ed_mask/spacing in context",
            }

        measurement_type = args.get("measurement_type", "lv_ef")

        if measurement_type in ("lv_ef", "rv_ef") and es_mask is not None:
            return self.measurement_tool.measure(
                measurement_type,
                ed_mask=ed_mask,
                es_mask=es_mask,
                spacing=spacing,
            )

        # When only ED mask is available (from a preceding segment call),
        # report a partial result with EDV only.
        if es_mask is None and ed_mask is not None:
            return self._partial_edv(ed_mask, spacing)

        return self.measurement_tool.measure(measurement_type, **context)

    @staticmethod
    def _partial_edv(
        ed_mask: np.ndarray,
        spacing: tuple[float, ...],
    ) -> dict[str, Any]:
        """Compute EDV-only partial result when ES mask is unavailable."""
        n_classes = 4
        arr = np.asarray(ed_mask)
        ed_oh = torch.nn.functional.one_hot(
            torch.from_numpy(arr).long(), n_classes,
        ).permute(-1, *range(arr.ndim)).unsqueeze(0).float()
        ed_vols = get_volumes(ed_oh, spacing)[0]
        return {
            "lv_edv_ml": round(float(ed_vols[LV_LABEL]), 1),
            "lv_esv_ml": None,
            "lv_ef_pct": None,
            "status": "partial",
            "message": "ES mask not available; only EDV reported.",
        }


__all__ = [
    "CardiacMeasurementTool",
    "CardiacSegmentationTool",
    "ToolRouter",
]
