"""Unified NIfTI loaders for cardiac MRI volumes.

Provides three loading modes:
- ``load_nifti_full``: SimpleITK-based, returns float array + spacing metadata.
- ``load_nifti_as_pil``: nibabel-based, returns PIL images for VLM processors.
- ``load_nifti_as_video_frames``: nibabel-based, returns frame list for video input.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk  # noqa: N813
from PIL import Image

from cardio.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 range [0, 255]."""
    arr = arr.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 0:
        arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)


def load_nifti_full(
    path: str | Path,
    frame_idx: int | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Load a NIfTI volume via SimpleITK, preserving spacing metadata.

    Args:
        path: Path to a ``.nii`` or ``.nii.gz`` file.
        frame_idx: If the volume is 4-D, extract this temporal frame (0-indexed).
            If *None* the full 4-D array is returned.

    Returns:
        A tuple ``(array, spacing)`` where *array* is a ``float32`` NumPy array
        in ``(X, Y, Z[, T])`` order and *spacing* is the per-axis spacing in mm
        from the image header.
    """
    image = sitk.ReadImage(str(path))
    ndim = image.GetDimension()

    if frame_idx is not None and ndim == 4:
        image = image[..., frame_idx]

    spacing = image.GetSpacing()
    # sitk.GetArrayFromImage returns (T, Z, Y, X) or (Z, Y, X); transpose
    # to the nibabel / CineMA convention (X, Y, Z[, T]).
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.transpose(arr)

    return arr, spacing


def load_nifti_as_pil(
    path: str | Path,
    slice_idx: int | None = None,
    frame_idx: int | None = None,
) -> list[Image.Image]:
    """Load a NIfTI volume via nibabel and return PIL images.

    Suitable for feeding into Qwen2.5-VL or similar VLM processors that expect
    ``PIL.Image`` inputs.

    Args:
        path: Path to a ``.nii`` or ``.nii.gz`` file.
        slice_idx: Spatial slice along the third axis to extract.
            If *None*, uses the mid-slice.
        frame_idx: For 4-D volumes, return only this temporal frame (wrapped in
            a single-element list).  If *None*, all frames are returned.

    Returns:
        A list of RGB ``PIL.Image`` objects -- one per temporal frame for 4-D
        data, or a single-element list for 3-D / 2-D data.
    """
    data = nib.load(str(path)).get_fdata()

    if data.ndim == 4:
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        sliced = data[:, :, slice_idx, :]  # (X, Y, T)

        if frame_idx is not None:
            return [
                Image.fromarray(_normalize_to_uint8(sliced[:, :, frame_idx])).convert("RGB"),
            ]
        return [
            Image.fromarray(_normalize_to_uint8(sliced[:, :, t])).convert("RGB")
            for t in range(sliced.shape[2])
        ]

    if data.ndim == 3:
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        return [
            Image.fromarray(_normalize_to_uint8(data[:, :, slice_idx])).convert("RGB"),
        ]

    # 2-D fallback
    return [Image.fromarray(_normalize_to_uint8(data)).convert("RGB")]


def load_nifti_as_video_frames(
    path: str | Path,
    target_fps: float | None = None,
) -> tuple[list[Image.Image], float]:
    """Load a 4-D NIfTI as a list of PIL frames for video-mode VLM input.

    Extracts the mid spatial slice across all temporal frames.

    Args:
        path: Path to a 4-D ``.nii`` or ``.nii.gz`` cine MRI file.
        target_fps: Desired output frame rate.  If provided and different from
            the native rate, the frame list is uniformly sub-sampled (or kept
            as-is when *target_fps* >= native fps) to approximate the target.

    Returns:
        A tuple ``(frames, fps)`` where *frames* is a list of RGB
        ``PIL.Image`` objects and *fps* is the effective frame rate.

    Raises:
        ValueError: If the file does not contain a 4-D volume.
    """
    nii = nib.load(str(path))
    data = nii.get_fdata()

    if data.ndim != 4:
        raise ValueError(
            f"load_nifti_as_video_frames requires a 4-D volume, got ndim={data.ndim}."
        )

    # Temporal spacing from header (seconds per frame)
    dt = float(nii.header["pixdim"][4])
    native_fps = 1.0 / dt if dt > 0 else 25.0

    mid_slice = data.shape[2] // 2
    n_frames = data.shape[3]

    # Build full frame list at native rate
    frame_indices = list(range(n_frames))

    if target_fps is not None and 0 < target_fps < native_fps and n_frames > 1:
        # Uniformly sub-sample to approximate target_fps
        keep = max(1, round(n_frames * target_fps / native_fps))
        frame_indices = [
            round(i * (n_frames - 1) / (keep - 1)) if keep > 1 else 0
            for i in range(keep)
        ]
        effective_fps = target_fps
    else:
        effective_fps = native_fps

    frames = [
        Image.fromarray(_normalize_to_uint8(data[:, :, mid_slice, t])).convert("RGB")
        for t in frame_indices
    ]
    return frames, effective_fps
