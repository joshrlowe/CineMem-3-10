"""Cardiac segmentation tool backend.

Wraps CineMA's :class:`~cardio.vision.ConvUNetR` segmentation model to
provide real inference behind the ``<tool_call>{"name": "segment_cardiac"}``
mechanism.  Loads finetuned weights from HuggingFace, runs per-frame
inference on NIfTI inputs, and returns segmentation masks with derived
measurements (areas, bounding boxes, volumes, ejection fractions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

from cardio.data.constants import LABEL_TO_NAME, LV_LABEL, MYO_LABEL, RV_LABEL
from cardio.data.io.nifti import load_nifti_full
from cardio.data.io.sitk import get_binary_mask_bounding_box
from cardio.utils.logging import get_logger
from cardio.vision import ConvUNetR
from cardio.vision.metric import ejection_fraction, get_volumes

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

_DEFAULT_REPO = "mathpluscode/CineMA"
_SAX_PAD_SIZE = (192, 192, 16)

_LABEL_NAMES = {v: k for k, v in LABEL_TO_NAME.items()}


def _mask_to_bbox_xyxy(mask_2d: np.ndarray) -> list[int]:
    """Convert a 2-D binary mask to ``[x1, y1, x2, y2]``.

    Returns ``[-1, -1, -1, -1]`` when the mask has no foreground.
    """
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask_2d)
    if int(bbox_min[0]) == -1:
        return [-1, -1, -1, -1]
    return [int(bbox_min[1]), int(bbox_min[0]), int(bbox_max[1]), int(bbox_max[0])]


class CardiacSegmentationTool:
    """Backend for ``<tool_call>{"name": "segment_cardiac"}`` invocations.

    Loads a CineMA ConvUNetR segmentation model finetuned on a cardiac
    dataset (ACDC/M&Ms) and exposes :meth:`segment` for single-frame
    inference and :meth:`compute_volumes` for ED/ES volume + EF
    computation.

    Args:
        repo_id: HuggingFace repository hosting the CineMA weights.
        dataset: training dataset identifier (e.g. ``"acdc"``).
        view: imaging view (``"sax"`` or ``"lax_4c"``).
        seed: model seed index within the HF repo (0-2).
        device: torch device string.
    """

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        dataset: str = "acdc",
        view: str = "sax",
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        """Initialise the segmentation model and transforms."""
        self.view = view
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        model_filename = (
            f"finetuned/segmentation/{dataset}_{view}/"
            f"{dataset}_{view}_{seed}.safetensors"
        )
        config_filename = f"finetuned/segmentation/{dataset}_{view}/config.yaml"

        self.model = ConvUNetR.from_finetuned(
            repo_id=repo_id,
            model_filename=model_filename,
            config_filename=config_filename,
        )
        self.model.to(self.device).eval()

        if view == "sax":
            self.transform = Compose([
                ScaleIntensityd(keys=view),
                SpatialPadd(keys=view, spatial_size=_SAX_PAD_SIZE, method="end"),
            ])
        else:
            self.transform = ScaleIntensityd(keys=view)

        logger.info(
            f"Loaded {dataset}_{view} segmentation model (seed={seed}) "
            f"on {self.device}."
        )

    # ------------------------------------------------------------------
    # Single-frame segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        nifti_path: str | Path,
        frame_idx: int | None = None,
    ) -> dict:
        """Run segmentation on a NIfTI volume.

        Args:
            nifti_path: path to ``.nii`` or ``.nii.gz`` file.
            frame_idx: temporal frame to segment.  If the input is 4-D
                and *frame_idx* is ``None``, frame 0 (typically ED) is
                used.

        Returns:
            Dict with keys ``mask``, ``spacing``, ``lv_area``,
            ``rv_area``, ``myo_area``, ``bbox_lv``, ``bbox_rv``.
        """
        array, spacing = load_nifti_full(str(nifti_path))

        if array.ndim == 4:
            frame_idx = frame_idx if frame_idx is not None else 0
            array = array[..., frame_idx]  # (X, Y, Z)

        n_slices = array.shape[-1]

        # images[None, ...] => (1, X, Y, Z) — single-channel batch
        inp = torch.from_numpy(array[None, ...])

        if self.view == "sax":
            batch = self.transform({self.view: inp})
            batch = {
                k: v[None, ...].to(device=self.device, dtype=torch.float32)
                for k, v in batch.items()
            }
        else:
            # LAX: drop Z dimension → (1, X, Y)
            if inp.ndim == 4 and inp.shape[-1] == 1:
                inp = inp[..., 0]
            batch = self.transform({self.view: inp})
            batch = {
                k: v[None, ...].to(device=self.device, dtype=torch.float32)
                for k, v in batch.items()
            }

        with (
            torch.no_grad(),
            torch.autocast(
                self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda",
            ),
        ):
            logits = self.model(batch)[self.view]  # (1, 4, X, Y[, Z])

        labels = torch.argmax(logits, dim=1)[0]  # (X, Y[, Z])

        if labels.ndim == 3:
            labels = labels[..., :n_slices]

        mask = labels.detach().cpu().float().numpy()

        lv_mask = mask == LV_LABEL
        rv_mask = mask == RV_LABEL
        myo_mask = mask == MYO_LABEL

        # Project to 2-D (max-projection along Z) for bounding boxes
        lv_proj = np.any(lv_mask, axis=-1) if lv_mask.ndim == 3 else lv_mask
        rv_proj = np.any(rv_mask, axis=-1) if rv_mask.ndim == 3 else rv_mask

        return {
            "mask": mask,
            "spacing": spacing,
            "lv_area": float(np.sum(lv_mask)),
            "rv_area": float(np.sum(rv_mask)),
            "myo_area": float(np.sum(myo_mask)),
            "bbox_lv": _mask_to_bbox_xyxy(lv_proj),
            "bbox_rv": _mask_to_bbox_xyxy(rv_proj),
        }

    # ------------------------------------------------------------------
    # Volume + EF from ED/ES masks
    # ------------------------------------------------------------------

    def compute_volumes(
        self,
        ed_mask: np.ndarray,
        es_mask: np.ndarray,
        spacing: tuple[float, ...],
    ) -> dict:
        """Compute ventricular volumes and ejection fractions.

        Args:
            ed_mask: end-diastolic label map ``(X, Y, Z)`` with values
                in ``{0, 1, 2, 3}`` (background / RV / MYO / LV).
            es_mask: end-systolic label map, same layout.
            spacing: voxel spacing in mm ``(sx, sy, sz)``.

        Returns:
            Dict with ``lv_edv``, ``lv_esv``, ``lv_ef``, ``rv_edv``,
            ``rv_esv``, ``rv_ef`` (volumes in mL, EF in %).
        """
        n_classes = 4  # background + RV + MYO + LV
        ed_onehot = torch.nn.functional.one_hot(
            torch.from_numpy(ed_mask).long(), n_classes,
        ).permute(-1, *range(ed_mask.ndim)).unsqueeze(0).float()
        es_onehot = torch.nn.functional.one_hot(
            torch.from_numpy(es_mask).long(), n_classes,
        ).permute(-1, *range(es_mask.ndim)).unsqueeze(0).float()

        ed_vols = get_volumes(ed_onehot, spacing)[0]  # (n_classes,)
        es_vols = get_volumes(es_onehot, spacing)[0]

        lv_edv = float(ed_vols[LV_LABEL])
        lv_esv = float(es_vols[LV_LABEL])
        rv_edv = float(ed_vols[RV_LABEL])
        rv_esv = float(es_vols[RV_LABEL])

        lv_ef = float(ejection_fraction(lv_edv, lv_esv)) if lv_edv > 0 else 0.0
        rv_ef = float(ejection_fraction(rv_edv, rv_esv)) if rv_edv > 0 else 0.0

        return {
            "lv_edv": round(lv_edv, 2),
            "lv_esv": round(lv_esv, 2),
            "lv_ef": round(lv_ef, 2),
            "rv_edv": round(rv_edv, 2),
            "rv_esv": round(rv_esv, 2),
            "rv_ef": round(rv_ef, 2),
        }
