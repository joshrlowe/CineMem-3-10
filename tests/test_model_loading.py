"""Tests for select_visual_positions (CineMemModel utilities)."""

from __future__ import annotations

import torch

from cardio.vlm.model import select_visual_positions

VISION_START_ID = 100
VISION_END_ID = 101


class TestSelectVisualPositions:
    def test_select_visual_positions_single_span(self) -> None:
        input_ids = torch.tensor([[1, 2, VISION_START_ID, 5, 5, 5, VISION_END_ID, 3, 4]])
        spans = select_visual_positions(input_ids, VISION_START_ID, VISION_END_ID)
        assert len(spans) == 1
        assert len(spans[0]) == 1
        start, end = spans[0][0]
        assert start == 2
        assert end == 6

    def test_select_visual_positions_multi_span(self) -> None:
        input_ids = torch.tensor([[
            1,
            VISION_START_ID, 5, 5, VISION_END_ID,
            2,
            VISION_START_ID, 5, 5, 5, VISION_END_ID,
            3,
        ]])
        spans = select_visual_positions(input_ids, VISION_START_ID, VISION_END_ID)
        assert len(spans) == 1
        assert len(spans[0]) == 2
        assert spans[0][0] == (1, 4)
        assert spans[0][1] == (6, 10)

    def test_select_visual_positions_no_vision(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        spans = select_visual_positions(input_ids, VISION_START_ID, VISION_END_ID)
        assert len(spans) == 1
        assert len(spans[0]) == 0
