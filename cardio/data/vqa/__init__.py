"""VQA generation pipeline for cardiac cine MRI datasets.

This module generates grounded Visual Question Answering (VQA) triplets
from ACDC, M&Ms, and M&Ms2 cardiac MRI datasets, using GPT-4o to produce
varied clinical reasoning grounded in verifiable patient metrics.

Extends the original CineMA discriminative categories (1-6) with multi-view
categories (7-12) and CineMem generative categories.
"""

from __future__ import annotations

from enum import Enum

from cardio.data.constants import LV_LABEL, MYO_LABEL, RV_LABEL

# ---------------------------------------------------------------------------
# Unified VQA category enum
# ---------------------------------------------------------------------------


class VQACategory(str, Enum):
    # CineMA discriminative (1-6)
    ABNORMALITY = "abnormality_localization"
    LV_FUNCTION = "lv_systolic_function"
    DILATION_LV = "ventricular_dilation_lv"
    DILATION_RV = "ventricular_dilation_rv"
    DIAGNOSIS = "diagnosis"
    BIVENTRICULAR = "biventricular_comparison"
    ED_ES = "ed_es_phase_comparison"
    # Multi-view schema (7-12)
    CROSS_VIEW_STRUCTURE = "cross_view_structure"
    MULTI_VIEW_CONTRACTION = "multi_view_contraction"
    RV_ASSESSMENT = "rv_assessment"
    MULTI_PHASE_SUMMARY = "multi_phase_functional_summary"
    CROSS_VIEW_CONSISTENCY = "cross_view_consistency_verification"
    TEMPORAL_CONTRACTION = "temporal_contraction_pattern"
    # CineMem generative
    BASIC = "basic"
    MULTI_STEP = "multi_step"
    TOOL_CALL = "tool_call"
    VERIFICATION = "verification"
    LONGITUDINAL = "longitudinal"
    MULTI_VIEW = "multi_view"


# ---------------------------------------------------------------------------
# Backward-compatible string aliases
# ---------------------------------------------------------------------------
CAT_ABNORMALITY = VQACategory.ABNORMALITY.value
CAT_LV_FUNCTION = VQACategory.LV_FUNCTION.value
CAT_DILATION_LV = VQACategory.DILATION_LV.value
CAT_DILATION_RV = VQACategory.DILATION_RV.value
CAT_DIAGNOSIS = VQACategory.DIAGNOSIS.value
CAT_BIVENTRICULAR = VQACategory.BIVENTRICULAR.value
CAT_ED_ES = VQACategory.ED_ES.value

ALL_CATEGORIES = [
    CAT_ABNORMALITY,
    CAT_LV_FUNCTION,
    CAT_DILATION_LV,
    CAT_DILATION_RV,
    CAT_DIAGNOSIS,
    CAT_BIVENTRICULAR,
    CAT_ED_ES,
]

# ---------------------------------------------------------------------------
# Multi-view input requirement mappings (Categories 7-12)
# ---------------------------------------------------------------------------
MULTI_VIEW_INPUTS: dict[VQACategory, list[str]] = {
    VQACategory.CROSS_VIEW_STRUCTURE: ["SAX_ED", "LAX_4CH_ED"],
    VQACategory.MULTI_VIEW_CONTRACTION: ["SAX_ED", "SAX_ES", "LAX_4CH_ED", "LAX_4CH_ES"],
    VQACategory.RV_ASSESSMENT: ["SAX_ED", "LAX_4CH_ED"],
    VQACategory.MULTI_PHASE_SUMMARY: ["SAX_ED", "SAX_ES", "LAX_4CH_ED", "LAX_4CH_ES"],
    VQACategory.CROSS_VIEW_CONSISTENCY: ["LAX_4CH_ED", "LAX_4CH_ES"],
    VQACategory.TEMPORAL_CONTRACTION: ["SAX_ED", "SAX_ES"],
}

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
DATASET_ACDC = "acdc"
DATASET_MNMS = "mnms"
DATASET_MNMS2 = "mnms2"

ALL_DATASETS = [DATASET_ACDC, DATASET_MNMS, DATASET_MNMS2]

# ---------------------------------------------------------------------------
# Pathologies to exclude from VQA generation
# ---------------------------------------------------------------------------
EXCLUDED_PATHOLOGIES = {"HHD", "TRI", "DRV"}

# ---------------------------------------------------------------------------
# Category 1: Abnormality Localization — answer maps
# ---------------------------------------------------------------------------
CAT1_QUESTION = (
    "Which cardiac structure shows the primary abnormality in this short-axis cine MRI?"
)
CAT1_OPTIONS = [
    {"id": "A", "text": "Left ventricle cavity -- appears dilated"},
    {"id": "B", "text": "Myocardium -- appears thickened"},
    {"id": "C", "text": "Right ventricle cavity -- appears dilated"},
    {"id": "D", "text": "Left ventricle -- shows reduced systolic function without significant dilation"},
    {"id": "E", "text": "No structural abnormality detected"},
]
CAT1_ANSWER_MAP: dict[str, str] = {
    "DCM": "A",
    "DLV": "A",
    "LV": "A",
    "HCM": "B",
    "ARV": "C",
    "RV": "C",
    "ARR": "C",
    "CIA": "C",
    "FALL": "C",
    "MINF": "D",
    "NOR": "E",
}
CAT1_GROUNDING_MAP: dict[str, tuple[str, int]] = {
    "A": ("LV", LV_LABEL),
    "B": ("MYO", MYO_LABEL),
    "C": ("RV", RV_LABEL),
    "D": ("LV", LV_LABEL),
    "E": ("NONE", -1),
}

# ---------------------------------------------------------------------------
# Category 2: LV Systolic Function — EF-based tiers
# ---------------------------------------------------------------------------
CAT2_QUESTION = (
    "Based on this short-axis cine MRI, how would you classify the left ventricular systolic function?"
)
CAT2_OPTIONS = [
    {"id": "A", "text": "Normal (EF > 55%)"},
    {"id": "B", "text": "Mildly reduced (41% <= EF <= 55%)"},
    {"id": "C", "text": "Moderately reduced (30% <= EF <= 40%)"},
    {"id": "D", "text": "Severely reduced (EF < 30%)"},
]


def get_cat2_answer(lv_ef: float) -> str:
    """Return the correct option letter for Category 2 given LV EF."""
    if lv_ef > 55:
        return "A"
    if lv_ef >= 41:
        return "B"
    if lv_ef >= 30:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Category 3: Ventricular Dilation — LV and RV
# ---------------------------------------------------------------------------
CAT3_LV_QUESTION = "Is the left ventricle dilated in this short-axis cine MRI?"
CAT3_RV_QUESTION = "Is the right ventricle dilated in this short-axis cine MRI?"
CAT3_OPTIONS = [
    {"id": "A", "text": "Yes -- the cavity appears significantly enlarged"},
    {"id": "B", "text": "No -- the cavity appears normal in size"},
]

CAT3_LV_DILATED: dict[str, str] = {
    "DCM": "A", "DLV": "A", "LV": "A",
    "HCM": "B", "MINF": "B",
    "ARV": "B", "RV": "B", "ARR": "B", "CIA": "B", "FALL": "B",
    "NOR": "B",
}
CAT3_RV_DILATED: dict[str, str] = {
    "DCM": "B", "DLV": "B", "LV": "B",
    "HCM": "B", "MINF": "B",
    "ARV": "A", "RV": "A", "ARR": "A", "CIA": "A", "FALL": "A",
    "NOR": "B",
}

# ---------------------------------------------------------------------------
# Category 4: Diagnosis — per-dataset option pools
# ---------------------------------------------------------------------------
CAT4_QUESTION = (
    "Based on this short-axis cine MRI, which diagnosis is most consistent with the observed findings?"
)

CAT4_ACDC_POOL = ["DCM", "HCM", "MINF", "ARV", "NOR"]
CAT4_MNMS_POOL = ["DCM", "HCM", "ARV", "NOR"]
CAT4_MNMS2_POOL = ["DLV", "HCM", "ARR", "FALL", "CIA", "NOR"]

CAT4_LABEL_TO_DISPLAY: dict[str, str] = {
    "DCM": "Dilated cardiomyopathy (DCM)",
    "DLV": "Dilated left ventricle (DLV)",
    "LV": "Dilated left ventricle (DLV)",
    "HCM": "Hypertrophic cardiomyopathy (HCM)",
    "MINF": "Myocardial infarction (MINF)",
    "ARV": "Abnormal right ventricle (ARV)",
    "RV": "Abnormal right ventricle (ARV)",
    "ARR": "Arrhythmogenic cardiomyopathy (ARR)",
    "FALL": "Tetralogy of Fallot (FALL)",
    "CIA": "Inter-atrial communication (CIA)",
    "NOR": "Normal cardiac function",
}

CAT4_GROUNDING_MAP: dict[str, tuple[str, int]] = {
    "DCM": ("LV", LV_LABEL),
    "DLV": ("LV", LV_LABEL),
    "LV": ("LV", LV_LABEL),
    "MINF": ("LV", LV_LABEL),
    "HCM": ("MYO", MYO_LABEL),
    "ARV": ("RV", RV_LABEL),
    "RV": ("RV", RV_LABEL),
    "ARR": ("RV", RV_LABEL),
    "FALL": ("RV", RV_LABEL),
    "CIA": ("RV", RV_LABEL),
    "NOR": ("NONE", -1),
}

# ---------------------------------------------------------------------------
# Category 5: Comparative Biventricular Assessment
# ---------------------------------------------------------------------------
CAT5_QUESTION = (
    "Comparing the left and right ventricles in this short-axis cine MRI, "
    "which statement is most accurate?"
)
CAT5_OPTIONS = [
    {"id": "A", "text": "The left ventricle is dilated; the right ventricle appears normal"},
    {"id": "B", "text": "The right ventricle is dilated; the left ventricle appears normal"},
    {"id": "C", "text": "Both ventricles appear dilated"},
    {"id": "D", "text": "Both ventricles appear normal in size and function"},
    {
        "id": "E",
        "text": (
            "The left ventricle shows reduced function but is not significantly dilated; "
            "the right ventricle appears normal"
        ),
    },
]
CAT5_ANSWER_MAP: dict[str, str] = {
    "DCM": "A", "DLV": "A", "LV": "A",
    "ARV": "B", "RV": "B", "ARR": "B", "CIA": "B", "FALL": "B",
    "NOR": "D", "HCM": "D",
    "MINF": "E",
}

# ---------------------------------------------------------------------------
# Category 6: ED vs ES Phase Comparison
# ---------------------------------------------------------------------------
CAT6_QUESTION = (
    "Comparing the end-diastole (ED) and end-systole (ES) frames of this "
    "short-axis cine MRI, what do you observe about left ventricular contraction?"
)
CAT6_OPTIONS = [
    {"id": "A", "text": "Normal contraction -- the LV cavity reduces significantly from ED to ES"},
    {"id": "B", "text": "Mildly impaired contraction -- the LV cavity reduces somewhat from ED to ES"},
    {"id": "C", "text": "Significantly impaired contraction -- the LV cavity barely changes from ED to ES"},
]


def get_cat6_answer(lv_ef: float) -> str:
    """Return the correct option letter for Category 6 given LV EF."""
    if lv_ef > 55:
        return "A"
    if lv_ef >= 41:
        return "B"
    return "C"
