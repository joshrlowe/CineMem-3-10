"""Tests for temporal ID generation (mRoPE fix)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from cardio.vlm.mrope_fix import TemporalAlignmentOverride


def _make_mock_base_model(
    tokens_per_second: float = 2.0,
    temporal_patch_size: int = 2,
) -> SimpleNamespace:
    """Construct a minimal mock base_model with config.vision_config."""
    vision_config = SimpleNamespace(
        tokens_per_second=tokens_per_second,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=2,
    )
    config = SimpleNamespace(vision_config=vision_config)
    model = SimpleNamespace(
        config=config,
        get_rope_index=lambda **kw: (torch.zeros(3, 1, 10), torch.zeros(1)),
    )
    return model


class TestDynamicTPSScaling:
    def test_dynamic_tps_scaling(self) -> None:
        model = _make_mock_base_model()
        override = TemporalAlignmentOverride(model)
        tps = override.compute_dynamic_tokens_per_second([30.0])
        assert tps == pytest.approx(45.0)

    def test_dynamic_tps_multiple_fps(self) -> None:
        model = _make_mock_base_model()
        override = TemporalAlignmentOverride(model)
        tps = override.compute_dynamic_tokens_per_second([10.0, 20.0, 30.0])
        assert tps == pytest.approx(45.0)

    def test_dynamic_tps_empty(self) -> None:
        model = _make_mock_base_model(tokens_per_second=2.0)
        override = TemporalAlignmentOverride(model)
        tps = override.compute_dynamic_tokens_per_second([])
        assert tps == pytest.approx(2.0)


class TestPhaseLockedIds:
    def test_phase_locked_ids_linear(self) -> None:
        model = _make_mock_base_model()
        override = TemporalAlignmentOverride(model)
        sax_ids, lax_ids = override.generate_phase_locked_temporal_ids(10, 10)
        expected = torch.arange(10, dtype=torch.long)
        assert torch.equal(sax_ids, expected)
        assert torch.equal(lax_ids, expected)

    def test_phase_locked_ids_custom_map(self) -> None:
        model = _make_mock_base_model()
        override = TemporalAlignmentOverride(model)
        rr_phase_map = {0: 0.0, 4: 0.35, 9: 1.0}
        sax_ids, _ = override.generate_phase_locked_temporal_ids(
            10, 10, rr_phase_map=rr_phase_map
        )
        # Frame 0 → phase 0.0 → tid 0
        assert sax_ids[0].item() == 0
        # Frame 4 → phase 0.35 → tid round(0.35 * 9) = round(3.15) = 3
        assert sax_ids[4].item() == 3
        # Frame 9 → phase 1.0 → tid round(1.0 * 9) = 9
        assert sax_ids[9].item() == 9


class TestUniformSpatialEnforcement:
    def test_uniform_spatial_enforcement(self) -> None:
        model = _make_mock_base_model()
        override = TemporalAlignmentOverride(model, default_total_pixels=1003520)
        video_inputs = [{"path": "sax.nii.gz"}, {"path": "lax.nii.gz"}]
        result = override.enforce_uniform_spatial(video_inputs)
        for v in result:
            assert v["total_pixels"] == 1003520
