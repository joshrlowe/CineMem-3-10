"""Medical image I/O loaders."""

from cardio.data.io.nifti import (
    load_nifti_as_pil,
    load_nifti_as_video_frames,
    load_nifti_full,
)

__all__ = [
    "load_nifti_as_pil",
    "load_nifti_as_video_frames",
    "load_nifti_full",
]
