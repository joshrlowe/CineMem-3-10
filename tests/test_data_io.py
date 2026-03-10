"""Tests for NIfTI loaders and image routing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cardio.data.collate import load_image
from cardio.data.io.nifti import load_nifti_as_pil


class TestLoadNiftiAsPil:
    def test_load_nifti_3d_returns_single_frame(self, sample_nifti_3d: Path) -> None:
        frames = load_nifti_as_pil(sample_nifti_3d)
        assert isinstance(frames, list)
        assert len(frames) == 1
        assert isinstance(frames[0], Image.Image)

    def test_load_nifti_4d_returns_frame_list(self, sample_nifti_4d: Path) -> None:
        frames = load_nifti_as_pil(sample_nifti_4d)
        assert isinstance(frames, list)
        assert len(frames) == 25

    def test_load_nifti_as_pil_shape(self, sample_nifti_4d: Path) -> None:
        frames = load_nifti_as_pil(sample_nifti_4d)
        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.mode == "RGB"
            w, h = frame.size
            assert w == 64
            assert h == 64


class TestLoadImage:
    def test_load_image_nifti_routing(self, sample_nifti_4d: Path) -> None:
        result = load_image(str(sample_nifti_4d))
        assert isinstance(result, list)
        assert len(result) == 25

    def test_load_image_none_returns_none(self) -> None:
        assert load_image(None) is None
