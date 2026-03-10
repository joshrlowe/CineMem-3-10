"""Shared fixtures for the CardioVLM test suite."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture()
def sample_nifti_3d(tmp_path: Path) -> Path:
    """64x64x10 random 3-D NIfTI volume."""
    data = np.random.default_rng(42).random((64, 64, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmp_path / "vol3d.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture()
def sample_nifti_4d(tmp_path: Path) -> Path:
    """64x64x10x25 random 4-D NIfTI cine (25 frames)."""
    data = np.random.default_rng(99).random((64, 64, 10, 25), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmp_path / "cine4d.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture()
def sample_reasoning_trace() -> str:
    """Realistic cardiac reasoning trace with embedded metrics."""
    return (
        "<think>"
        "The patient presents with dyspnea on exertion. "
        "Echocardiographic assessment reveals LVEF is 55%, consistent with "
        "preserved systolic function. LAVI of 38 mL/m2 indicates left "
        "atrial enlargement. E/e' is 12, suggesting mildly elevated filling "
        "pressures. GLS of -18% is borderline. "
        "Given the preserved EF with structural evidence of diastolic "
        "dysfunction, the diagnosis is consistent with HFpEF."
        "</think>"
        "The findings are consistent with Heart Failure with Preserved "
        "Ejection Fraction (HFpEF)."
    )


@pytest.fixture()
def sample_invocation_log() -> list[dict]:
    """Simulated memory invocation log with TDM and PSM entries."""
    return [
        {"step": 10, "token_idx": 10, "type": "tdm"},
        {"step": 30, "token_idx": 30, "type": "psm"},
        {"step": 55, "token_idx": 55, "type": "tdm"},
        {"step": 80, "token_idx": 80, "type": "psm"},
        {"step": 120, "token_idx": 120, "type": "tdm"},
    ]
