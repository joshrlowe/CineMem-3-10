"""Patch Qwen2.5-VL temporal position encoding (MRoPE fix).

Addresses `QwenLM/Qwen2.5-VL#1031`_ and `#893`_: when
``tokens_per_second`` is low (default 2) and the video FPS exceeds
``tokens_per_second * temporal_patch_size / 2``, the temporal tensor
generation produces fractional values that truncate to identical integers
via ``.long()`` casting.  Multiple consecutive frames then share the same
temporal position ID, destroying phase synchronisation.

The :class:`TemporalAlignmentOverride` monkey-patches ``get_rope_index``
on the Qwen2.5-VL model with a thin closure wrapper that:

1. **Dynamic FPS scaling** — raises ``tokens_per_second`` so every
   temporal grid step maps to a distinct integer.
2. **Uniform spatial constraints** — forces all interleaved video inputs
   to identical ``total_pixels`` so temporal strides match after the 3-D
   convolution.
3. **Phase-locked embedding override** — optionally overrides the
   sequential temporal tensor with cardiac-phase-aligned IDs derived
   from an ECG R-R interval map.

.. _QwenLM/Qwen2.5-VL#1031: https://github.com/QwenLM/Qwen2.5-VL/issues/1031
.. _#893: https://github.com/QwenLM/Qwen2.5-VL/issues/893
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn as nn

from cardio.utils.logging import get_logger

logger = get_logger(__name__)


class TemporalAlignmentOverride:
    """Patch the Qwen2.5-VL vision encoder's temporal position computation.

    Must be applied **before** any forward pass or generation call.

    Args:
        base_model: the ``Qwen2_5_VLForConditionalGeneration`` instance.
        default_total_pixels: uniform ``total_pixels`` enforced on all
            video inputs so that SAX and LAX share temporal strides.
    """

    def __init__(
        self,
        base_model: nn.Module,
        default_total_pixels: int = 1003520,
    ) -> None:
        self.base_model = base_model
        self.default_total_pixels = default_total_pixels

        vision_cfg = getattr(base_model.config, "vision_config", None)
        if vision_cfg is None:
            msg = "base_model.config has no vision_config attribute."
            raise ValueError(msg)

        self._temporal_patch_size: int = getattr(vision_cfg, "temporal_patch_size", 2)
        self._original_tps: float = float(getattr(vision_cfg, "tokens_per_second", 2))
        self._original_get_rope_index = base_model.get_rope_index
        self._rr_phase_map: dict[int, float] | None = None
        self._patched = False

    # ------------------------------------------------------------------
    # Fix 1: Dynamic FPS scaling
    # ------------------------------------------------------------------

    def compute_dynamic_tokens_per_second(
        self,
        fps_list: list[float],
    ) -> float:
        """Return a ``tokens_per_second`` that prevents rounding collisions.

        The value is ``max(fps_list) * 1.5``, which guarantees that the
        per-grid time step ``second_per_grid_t * tps`` is >= 1.5 for the
        fastest video, so ``.long()`` truncation never maps two
        consecutive temporal grids to the same integer.
        """
        if not fps_list:
            return self._original_tps
        return max(fps_list) * 1.5

    # ------------------------------------------------------------------
    # Fix 2: Uniform spatial constraints
    # ------------------------------------------------------------------

    def enforce_uniform_spatial(
        self,
        video_inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Set ``total_pixels`` on every video input dict to the default.

        This ensures all interleaved videos (SAX and LAX) produce
        identical ``llm_grid_h * llm_grid_w`` after the processor's
        spatial resizing, so their temporal strides match after the 3-D
        convolution.
        """
        for v in video_inputs:
            v["total_pixels"] = self.default_total_pixels
        return video_inputs

    # ------------------------------------------------------------------
    # Fix 3: Phase-locked temporal IDs
    # ------------------------------------------------------------------

    def set_phase_map(self, rr_phase_map: dict[int, float]) -> None:
        """Register a cardiac R-R phase mapping for temporal overrides.

        Args:
            rr_phase_map: maps frame index to cardiac phase percentage
                (``0.0`` = R-wave, ``~0.35`` = end-systole,
                ``1.0`` = next R-wave).
        """
        self._rr_phase_map = rr_phase_map

    def generate_phase_locked_temporal_ids(
        self,
        n_frames_sax: int,
        n_frames_lax: int,
        rr_phase_map: dict[int, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synchronised temporal IDs for SAX and LAX sequences.

        If *rr_phase_map* is provided, frame indices are mapped to
        cardiac-phase percentages and quantised to integer temporal IDs.
        Frames at the same cardiac phase (e.g. end-systole at ~35 % of
        R-R) receive identical temporal IDs across views.

        When *rr_phase_map* is ``None``, a default linear mapping
        ``phase[i] = i / n_frames`` is used.

        Args:
            n_frames_sax: number of SAX temporal frames after 3-D conv.
            n_frames_lax: number of LAX temporal frames after 3-D conv.
            rr_phase_map: optional frame-index-to-phase dict.

        Returns:
            ``(sax_temporal_ids, lax_temporal_ids)`` — integer tensors of
            length *n_frames_sax* and *n_frames_lax* respectively.
        """
        pm = rr_phase_map or self._rr_phase_map
        max_n = max(n_frames_sax, n_frames_lax)
        # Scale factor: largest frame count determines the ID range
        max_tid = max_n - 1 if max_n > 1 else 1

        def _ids_for(n: int) -> torch.Tensor:
            ids = torch.zeros(n, dtype=torch.long)
            for i in range(n):
                if pm is not None and i in pm:
                    phase = pm[i]
                else:
                    phase = i / max(n - 1, 1)
                ids[i] = round(phase * max_tid)
            return ids

        return _ids_for(n_frames_sax), _ids_for(n_frames_lax)

    # ------------------------------------------------------------------
    # Monkey-patch
    # ------------------------------------------------------------------

    def patch_model(self) -> None:
        """Monkey-patch ``get_rope_index`` on the base model.

        The patched method wraps the original:

        1. Recovers per-video FPS from ``second_per_grid_ts`` and
           temporarily raises ``tokens_per_second`` to prevent rounding
           collisions.
        2. Delegates to the original ``get_rope_index``.
        3. Restores ``tokens_per_second``.
        4. (Optional) Overwrites the temporal axis of ``position_ids``
           with phase-locked cardiac IDs when a R-R phase map is active.
        """
        if self._patched:
            return

        original = self._original_get_rope_index
        override = self  # captured by the closure

        @functools.wraps(original)
        def _patched_get_rope_index(
            input_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=None,
        ):
            vision_cfg = override.base_model.config.vision_config
            saved_tps = vision_cfg.tokens_per_second

            # --- Fix 1: dynamic TPS ---
            if second_per_grid_ts is not None:
                tps = override._temporal_patch_size
                fps_list: list[float] = []
                for s in second_per_grid_ts:
                    val = s.item() if isinstance(s, torch.Tensor) else float(s)
                    if val > 0:
                        fps_list.append(tps / val)
                if fps_list:
                    required_tps = override.compute_dynamic_tokens_per_second(fps_list)
                    vision_cfg.tokens_per_second = required_tps

            try:
                position_ids, mrope_deltas = original(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
            finally:
                vision_cfg.tokens_per_second = saved_tps

            # --- Fix 3: phase-locked override ---
            if (
                override._rr_phase_map is not None
                and video_grid_thw is not None
                and input_ids is not None
            ):
                position_ids = override._apply_phase_override(
                    position_ids, input_ids, video_grid_thw, attention_mask,
                )

            return position_ids, mrope_deltas

        self.base_model.get_rope_index = _patched_get_rope_index
        self._patched = True
        logger.info(
            "MRoPE temporal fix applied (original tps=%.1f, temporal_patch_size=%d)",
            self._original_tps,
            self._temporal_patch_size,
        )

    def unpatch_model(self) -> None:
        """Restore the original ``get_rope_index``."""
        if not self._patched:
            return
        # Remove the instance-level override so the class method is
        # visible again.  Fall back to explicit reassignment if the
        # attribute was not set on the instance (should not happen).
        try:
            delattr(self.base_model, "get_rope_index")
        except AttributeError:
            self.base_model.get_rope_index = self._original_get_rope_index
        self.base_model.config.vision_config.tokens_per_second = self._original_tps
        self._patched = False
        logger.info("MRoPE temporal fix removed, tps restored to %.1f", self._original_tps)

    # ------------------------------------------------------------------
    # Phase override internals
    # ------------------------------------------------------------------

    def _apply_phase_override(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        video_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Overwrite the temporal axis of video spans with phase-locked IDs.

        Locates video token spans in *input_ids* (via
        ``vision_start_token_id`` / ``video_token_id``), then replaces
        ``position_ids[0, batch, span]`` with phase-locked temporal IDs
        repeated over the spatial grid.
        """
        cfg = self.base_model.config
        vs_id = getattr(cfg, "vision_start_token_id", None)
        vid_id = getattr(cfg, "video_token_id", None)
        if vs_id is None or vid_id is None:
            return position_ids

        spatial_merge = getattr(cfg.vision_config, "spatial_merge_size", 2)
        position_ids = position_ids.clone()

        video_idx = 0
        for b in range(input_ids.size(0)):
            row = input_ids[b]
            if attention_mask is not None:
                row = row[attention_mask[b] == 1]

            vs_positions = torch.where(row == vs_id)[0]
            for vs_pos in vs_positions:
                if vs_pos + 1 >= len(row) or row[vs_pos + 1] != vid_id:
                    continue
                if video_idx >= video_grid_thw.size(0):
                    break

                t, h, w = video_grid_thw[video_idx].tolist()
                llm_h = h // spatial_merge
                llm_w = w // spatial_merge
                n_vis_tokens = t * llm_h * llm_w

                sax_ids, lax_ids = self.generate_phase_locked_temporal_ids(t, t)
                phase_ids = sax_ids if video_idx == 0 else lax_ids

                # Build the temporal ID pattern: each frame's ID repeated
                # over the spatial grid (llm_h * llm_w tokens per frame)
                t_override = phase_ids.repeat_interleave(llm_h * llm_w)

                # Find the span start in the *original* (possibly padded)
                # input_ids; the video tokens start right after
                # <|vision_start|> <|video|>
                orig_row = input_ids[b]
                all_vs = torch.where(orig_row == vs_id)[0]
                # Match the video_idx-th video-type vision_start
                vid_count = 0
                for vsp in all_vs:
                    if vsp + 1 < orig_row.size(0) and orig_row[vsp + 1] == vid_id:
                        if vid_count == video_idx:
                            span_start = vsp.item() + 1  # after <|vision_start|>
                            span_end = span_start + n_vis_tokens
                            if span_end <= position_ids.size(2):
                                base_offset = position_ids[0, b, span_start].item()
                                position_ids[0, b, span_start : span_end] = (
                                    t_override[: span_end - span_start].to(position_ids.device)
                                    + base_offset
                                )
                            break
                        vid_count += 1

                video_idx += 1

        return position_ids
