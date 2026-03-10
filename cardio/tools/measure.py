"""Cardiac measurement tool backend.

Provides the ``<tool_call>{"name": "measure_volume"}`` backend and
additional clinical measurement routines that operate on pre-computed
segmentation masks.  All methods are stateless and return dicts whose
keys match the ``<tool_result>`` JSON format used in VQA training data.
"""

from __future__ import annotations

import numpy as np
import torch

from cardio.data.constants import LV_LABEL, RV_LABEL
from cardio.utils.logging import get_logger
from cardio.vision.metric import ejection_fraction, get_volumes

logger = get_logger(__name__)


def _one_hot_from_labels(
    mask: np.ndarray,
    n_classes: int = 4,
) -> torch.Tensor:
    """Convert a label map to a ``(1, C, ...)`` one-hot float tensor."""
    t = torch.from_numpy(mask).long()
    oh = torch.nn.functional.one_hot(t, n_classes)  # (..., C)
    # Move class dim to front and add batch dim
    oh = oh.permute(-1, *range(mask.ndim)).unsqueeze(0).float()
    return oh


class CardiacMeasurementTool:
    """Backend for ``<tool_call>{"name": "measure_volume"}`` and friends.

    All methods accept NumPy arrays (label maps or binary masks) plus
    spacing metadata and return measurement dicts.
    """

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def measure(self, measurement_type: str, **kwargs: object) -> dict:
        """Route to a specific measurement function.

        Args:
            measurement_type: one of ``"lv_ef"``, ``"rv_ef"``,
                ``"lavi"``, ``"wall_thickness"``.
            **kwargs: forwarded to the concrete method.

        Returns:
            Measurement dict with named float values.

        Raises:
            ValueError: if *measurement_type* is not recognised.
        """
        dispatch = {
            "lv_ef": self._compute_lv_ef,
            "rv_ef": self._compute_rv_ef,
            "lavi": self._compute_lavi,
            "wall_thickness": self._compute_wall_thickness,
        }
        fn = dispatch.get(measurement_type)
        if fn is None:
            msg = (
                f"Unknown measurement_type '{measurement_type}'. "
                f"Supported: {sorted(dispatch)}"
            )
            raise ValueError(msg)
        return fn(**kwargs)

    # ------------------------------------------------------------------
    # Ejection fraction helpers
    # ------------------------------------------------------------------

    def _compute_lv_ef(
        self,
        ed_mask: np.ndarray,
        es_mask: np.ndarray,
        spacing: tuple[float, ...],
        **_: object,
    ) -> dict:
        """Compute LV volumes and ejection fraction.

        Args:
            ed_mask: end-diastolic label map ``(X, Y, Z)``.
            es_mask: end-systolic label map ``(X, Y, Z)``.
            spacing: voxel spacing in mm.
        """
        ed_oh = _one_hot_from_labels(ed_mask)
        es_oh = _one_hot_from_labels(es_mask)
        ed_vols = get_volumes(ed_oh, spacing)[0]
        es_vols = get_volumes(es_oh, spacing)[0]

        lv_edv = float(ed_vols[LV_LABEL])
        lv_esv = float(es_vols[LV_LABEL])
        lv_ef = float(ejection_fraction(lv_edv, lv_esv)) if lv_edv > 0 else 0.0

        return {
            "lv_edv_ml": round(lv_edv, 1),
            "lv_esv_ml": round(lv_esv, 1),
            "lv_ef_pct": round(lv_ef, 1),
        }

    def _compute_rv_ef(
        self,
        ed_mask: np.ndarray,
        es_mask: np.ndarray,
        spacing: tuple[float, ...],
        **_: object,
    ) -> dict:
        """Compute RV volumes and ejection fraction."""
        ed_oh = _one_hot_from_labels(ed_mask)
        es_oh = _one_hot_from_labels(es_mask)
        ed_vols = get_volumes(ed_oh, spacing)[0]
        es_vols = get_volumes(es_oh, spacing)[0]

        rv_edv = float(ed_vols[RV_LABEL])
        rv_esv = float(es_vols[RV_LABEL])
        rv_ef = float(ejection_fraction(rv_edv, rv_esv)) if rv_edv > 0 else 0.0

        return {
            "rv_edv_ml": round(rv_edv, 1),
            "rv_esv_ml": round(rv_esv, 1),
            "rv_ef_pct": round(rv_ef, 1),
        }

    # ------------------------------------------------------------------
    # Left atrial volume index
    # ------------------------------------------------------------------

    def _compute_lavi(
        self,
        la_mask: np.ndarray,
        spacing: tuple[float, ...],
        bsa: float,
        **_: object,
    ) -> dict:
        """Compute left atrial volume index (LAVI).

        LAVI = LA_volume / BSA, where LA_volume is in mL and BSA in m^2.

        Args:
            la_mask: binary mask of the left atrium ``(X, Y, Z)``.
            spacing: voxel spacing in mm.
            bsa: body surface area in m^2.
        """
        voxel_vol_mm3 = float(np.prod(spacing))
        la_vol_ml = float(np.sum(la_mask > 0)) * voxel_vol_mm3 / 1000.0

        lavi = la_vol_ml / bsa if bsa > 0 else 0.0

        return {
            "la_vol_ml": round(la_vol_ml, 1),
            "bsa_m2": round(bsa, 2),
            "lavi_ml_m2": round(lavi, 1),
        }

    # ------------------------------------------------------------------
    # Mean wall thickness
    # ------------------------------------------------------------------

    def _compute_wall_thickness(
        self,
        myo_mask: np.ndarray,
        spacing: tuple[float, ...],
        **_: object,
    ) -> dict:
        """Estimate mean myocardial wall thickness from a MYO binary mask.

        Uses the area-perimeter approach: for each slice, computes the
        endocardial and epicardial perimeters and derives mean thickness
        as ``myo_area / mean_perimeter``.  Results are averaged over
        slices.

        Args:
            myo_mask: binary mask of the myocardium ``(X, Y, Z)`` or
                ``(X, Y)``.
            spacing: voxel spacing in mm (at least 2 elements for X, Y).
        """
        from scipy import ndimage

        pixel_area = float(spacing[0] * spacing[1])

        slices = (
            [myo_mask]
            if myo_mask.ndim == 2
            else [myo_mask[:, :, z] for z in range(myo_mask.shape[2])]
        )

        thicknesses: list[float] = []
        for sl in slices:
            binary = sl > 0
            if not np.any(binary):
                continue

            myo_area_mm2 = float(np.sum(binary)) * pixel_area

            struct = ndimage.generate_binary_structure(2, 1)
            eroded = ndimage.binary_erosion(binary, structure=struct)
            dilated = ndimage.binary_dilation(binary, structure=struct)
            endo_perimeter = float(np.sum(binary & ~eroded))
            epi_perimeter = float(np.sum(dilated & ~binary))
            mean_perimeter = (endo_perimeter + epi_perimeter) / 2.0

            if mean_perimeter > 0:
                thicknesses.append(myo_area_mm2 / mean_perimeter)

        mean_thickness = float(np.mean(thicknesses)) if thicknesses else 0.0

        return {
            "mean_wall_thickness_mm": round(mean_thickness, 2),
            "n_slices_measured": len(thicknesses),
        }
