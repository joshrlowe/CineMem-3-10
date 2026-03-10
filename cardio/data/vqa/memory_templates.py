"""Generate memory-token-annotated VQA training data.

Ports CineMem's build_vqa.py logic for producing five generative categories
(basic, multi_step, tool_call, verification, longitudinal) and four multi-view
categories, each annotated with short-term (<ms_I><ms_E>) and long-term
(<ml_I><ml_E>) memory invocation tokens.

Also provides ``annotate_ground_truth`` for wrapping CineMA discriminative
ground-truth entries with ``<think>``/``<answer>`` tags and memory tokens.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Memory-invocation tokens (single source of truth in cardio.vlm.constants)
# ---------------------------------------------------------------------------
from cardio.vlm.constants import LONG_END as ML_E
from cardio.vlm.constants import LONG_INVOKE as ML_I
from cardio.vlm.constants import SHORT_END as MS_E
from cardio.vlm.constants import SHORT_INVOKE as MS_I

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class VQASample:
    id: str
    image: str | None
    prompt: str
    answer: str
    category: str
    dataset: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ef_severity(ef: float) -> str:
    if ef >= 55:
        return "normal"
    if ef >= 45:
        return "mildly reduced"
    if ef >= 35:
        return "moderately reduced"
    return "severely reduced"


def _fmt(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}"


def _maybe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. BASIC -- direct Q&A
# ---------------------------------------------------------------------------


def _basic_ef(row: dict, ds: str, view_col: str) -> list[VQASample]:
    """EF question from LV or RV volumes."""
    samples = []

    for side, edv_col, esv_col, ef_col in [
        ("left ventricular", "lv_edv", "lv_esv", "lv_ef"),
        ("right ventricular", "rv_edv", "rv_esv", "rv_ef"),
    ]:
        ef = _maybe_float(row.get(ef_col))
        edv = _maybe_float(row.get(edv_col))
        esv = _maybe_float(row.get(esv_col))
        if ef is None:
            continue

        short_side = "LV" if "left" in side else "RV"
        pid = row["pid"]
        samples.append(VQASample(
            id=f"{ds}_{pid}_basic_{short_side}_ef",
            image=row.get(view_col),
            prompt=f"What is the {side} ejection fraction for this cardiac MRI?",
            answer=f"The {side} ejection fraction ({short_side} EF) is {_fmt(ef)}%, which is {_ef_severity(ef)}.",
            category="basic",
            dataset=ds,
            meta={"ef": ef, "edv": edv, "esv": esv, "side": short_side},
        ))
    return samples


def _basic_pathology(row: dict, ds: str, view_col: str) -> list[VQASample]:
    pathology = row.get("pathology")
    if not pathology or str(pathology) == "nan":
        return []
    pid = row["pid"]

    path_map = {
        "NOR": "normal cardiac function",
        "DCM": "dilated cardiomyopathy",
        "HCM": "hypertrophic cardiomyopathy",
        "MINF": "prior myocardial infarction",
        "ARV": "arrhythmogenic right ventricular cardiomyopathy",
        "N": "normal cardiac morphology",
        "P": "pathological findings consistent with myocardial infarction",
    }
    desc = path_map.get(str(pathology), str(pathology))

    return [VQASample(
        id=f"{ds}_{pid}_basic_pathology",
        image=row.get(view_col),
        prompt="What is the cardiac diagnosis for this patient based on the cine-MRI?",
        answer=f"The diagnosis is **{pathology}** — {desc}.",
        category="basic",
        dataset=ds,
        meta={"pathology": str(pathology)},
    )]


def _basic_volume(row: dict, ds: str, view_col: str) -> list[VQASample]:
    edv = _maybe_float(row.get("lv_edv") or row.get("diastole_volume"))
    esv = _maybe_float(row.get("lv_esv") or row.get("systole_volume"))
    if edv is None and esv is None:
        return []
    pid = row["pid"]

    parts = []
    if edv is not None:
        parts.append(f"LV end-diastolic volume (EDV) = {_fmt(edv)} mL")
    if esv is not None:
        parts.append(f"LV end-systolic volume (ESV) = {_fmt(esv)} mL")

    return [VQASample(
        id=f"{ds}_{pid}_basic_volume",
        image=row.get(view_col),
        prompt="Report the left ventricular volumes measured from this cardiac MRI.",
        answer=". ".join(parts) + ".",
        category="basic",
        dataset=ds,
        meta={"edv": edv, "esv": esv},
    )]


# ---------------------------------------------------------------------------
# 2. MULTI-STEP -- chain-of-thought reasoning with memory tokens
# ---------------------------------------------------------------------------


def _multistep_ef_analysis(row: dict, ds: str, view_col: str) -> list[VQASample]:
    ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
    edv = _maybe_float(row.get("lv_edv") or row.get("diastole_volume"))
    esv = _maybe_float(row.get("lv_esv") or row.get("systole_volume"))
    if ef is None or edv is None or esv is None:
        return []
    pid = row["pid"]
    pathology = row.get("pathology", "unknown")
    severity = _ef_severity(ef)

    answer = (
        f"<step>OBSERVE</step>\n"
        f"I examine the short-axis cine-MRI. "
    )
    if ef < 45:
        answer += "The left ventricular cavity appears dilated with reduced wall thickening during systole."
    elif ef > 65:
        answer += "The left ventricle shows normal dimensions with vigorous wall motion."
    else:
        answer += "The left ventricle shows borderline dimensions with mildly reduced contraction."

    answer += (
        f"\n{MS_I}{MS_E}\n\n"
        f"<step>MEASURE</step>\n"
        f"End-diastolic volume (EDV) = {_fmt(edv)} mL\n"
        f"End-systolic volume (ESV) = {_fmt(esv)} mL\n\n"
        f"<step>REASON</step>\n"
        f"Ejection Fraction = (EDV \u2212 ESV) / EDV \u00d7 100%\n"
        f"EF = ({_fmt(edv)} \u2212 {_fmt(esv)}) / {_fmt(edv)} \u00d7 100% = {_fmt(ef)}%\n"
        f"An EF of {_fmt(ef)}% is classified as **{severity}**"
    )
    if str(pathology) not in ("nan", "unknown", "None"):
        answer += f", consistent with {pathology}"
    answer += "."

    answer += (
        f"\n{ML_I}{ML_E}\n\n"
        f"<step>CONCLUDE</step>\n"
        f"The LV ejection fraction is {_fmt(ef)}% ({severity}). "
    )
    sv = edv - esv
    answer += f"Stroke volume is {_fmt(sv)} mL."

    return [VQASample(
        id=f"{ds}_{pid}_multistep_ef",
        image=row.get(view_col),
        prompt=(
            "Analyze this cardiac cine-MRI step by step. "
            "Observe the cardiac structures, measure volumes, "
            "calculate ejection fraction, and provide your assessment."
        ),
        answer=answer,
        category="multi_step",
        dataset=ds,
        meta={"ef": ef, "edv": edv, "esv": esv, "pathology": str(pathology)},
    )]


def _multistep_morphology(row: dict, ds: str, view_col: str) -> list[VQASample]:
    """Multi-step morphology assessment (for datasets with n_slices, spacing)."""
    n_slices = _maybe_float(row.get("n_slices"))
    if n_slices is None:
        return []
    pid = row["pid"]
    pathology = row.get("pathology", "unknown")
    n_slices = int(n_slices)

    sz = _maybe_float(row.get("original_sax_spacing_z") or row.get("orig_sax_spacing_z") or row.get("orig_spacing_z"))
    coverage = f"{_fmt(n_slices * sz)} mm" if sz else f"{n_slices} slices"

    answer = (
        f"<step>OBSERVE</step>\n"
        f"The short-axis stack contains {n_slices} slices covering approximately {coverage} "
        f"from base to apex.\n"
        f"{MS_I}{MS_E}\n\n"
        f"<step>ASSESS</step>\n"
    )
    if str(pathology) not in ("nan", "unknown", "None"):
        answer += f"Clinical label: {pathology}. "

    ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
    if ef is not None:
        answer += f"LV EF = {_fmt(ef)}% ({_ef_severity(ef)}). "

    answer += (
        f"\n{ML_I}{ML_E}\n\n"
        f"<step>CONCLUDE</step>\n"
        f"The cardiac morphology assessment is complete for this {n_slices}-slice SAX stack."
    )

    return [VQASample(
        id=f"{ds}_{pid}_multistep_morphology",
        image=row.get(view_col),
        prompt=(
            "Perform a structured morphology assessment of this cardiac MRI. "
            "Describe the scan coverage, assess cardiac structures, and summarize findings."
        ),
        answer=answer,
        category="multi_step",
        dataset=ds,
        meta={"n_slices": n_slices, "pathology": str(pathology)},
    )]


# ---------------------------------------------------------------------------
# 3. TOOL CALL -- segmentation / measurement tool with controller loop
# ---------------------------------------------------------------------------

TOOL_SEGMENT = "segment_cardiac"
TOOL_MEASURE = "measure_volume"


def _tool_segment_and_measure(row: dict, ds: str, view_col: str) -> list[VQASample]:
    ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
    edv = _maybe_float(row.get("lv_edv") or row.get("diastole_volume"))
    esv = _maybe_float(row.get("lv_esv") or row.get("systole_volume"))
    if ef is None or edv is None or esv is None:
        return []
    pid = row["pid"]
    rv_edv = _maybe_float(row.get("rv_edv"))
    rv_esv = _maybe_float(row.get("rv_esv"))

    seg_result = {"structures": ["LV", "RV", "MYO"], "status": "success"}
    measure_result = {"lv_edv_ml": round(edv, 1), "lv_esv_ml": round(esv, 1), "lv_ef_pct": round(ef, 1)}
    if rv_edv is not None:
        measure_result["rv_edv_ml"] = round(rv_edv, 1)
    if rv_esv is not None:
        measure_result["rv_esv_ml"] = round(rv_esv, 1)

    answer = (
        f"I will segment the cardiac structures and measure volumes.\n\n"
        f'<tool_call>{{"name": "{TOOL_SEGMENT}", "args": {{"image": "sax_ed", "structures": ["LV", "RV", "MYO"]}}}}</tool_call>\n'
        f"<tool_result>{json.dumps(seg_result)}</tool_result>\n\n"
        f"Segmentation successful. Now measuring volumes.\n"
        f"{MS_I}{MS_E}\n\n"
        f'<tool_call>{{"name": "{TOOL_MEASURE}", "args": {{"segmentation": "sax_ed_gt", "spacing": "auto"}}}}</tool_call>\n'
        f"<tool_result>{json.dumps(measure_result)}</tool_result>\n\n"
        f"Measurements complete:\n"
        f"- LV EDV = {_fmt(edv)} mL\n"
        f"- LV ESV = {_fmt(esv)} mL\n"
        f"- LV EF = {_fmt(ef)}% ({_ef_severity(ef)})\n"
    )
    if rv_edv is not None and rv_esv is not None:
        rv_ef = (rv_edv - rv_esv) / rv_edv * 100 if rv_edv > 0 else 0
        answer += f"- RV EDV = {_fmt(rv_edv)} mL, RV ESV = {_fmt(rv_esv)} mL, RV EF = {_fmt(rv_ef)}%\n"

    answer += f"\n{ML_I}{ML_E}"

    return [VQASample(
        id=f"{ds}_{pid}_tool_segment_measure",
        image=row.get(view_col),
        prompt=(
            "Segment the cardiac structures in this cine-MRI and measure "
            "the ventricular volumes. Use available tools to compute results."
        ),
        answer=answer,
        category="tool_call",
        dataset=ds,
        meta={"tools_used": [TOOL_SEGMENT, TOOL_MEASURE]},
    )]


# ---------------------------------------------------------------------------
# 4. VERIFICATION -- self-check + consistency check
# ---------------------------------------------------------------------------


def _verify_ef(row: dict, ds: str, view_col: str) -> list[VQASample]:
    ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
    edv = _maybe_float(row.get("lv_edv") or row.get("diastole_volume"))
    esv = _maybe_float(row.get("lv_esv") or row.get("systole_volume"))
    if ef is None or edv is None or esv is None:
        return []
    pid = row["pid"]
    pathology = row.get("pathology", "unknown")
    recomputed_ef = (edv - esv) / edv * 100 if edv > 0 else 0
    ef_diff = abs(ef - recomputed_ef)

    rv_edv = _maybe_float(row.get("rv_edv"))
    rv_ef = None
    if rv_edv is not None:
        rv_esv = _maybe_float(row.get("rv_esv"))
        if rv_esv is not None and rv_edv > 0:
            rv_ef = (rv_edv - rv_esv) / rv_edv * 100

    answer = (
        f"<step>INITIAL ASSESSMENT</step>\n"
        f"LV EDV = {_fmt(edv)} mL, ESV = {_fmt(esv)} mL, EF = {_fmt(ef)}%\n"
        f"{MS_I}{MS_E}\n\n"
        f"<verify>\n"
        f"SELF-CHECK \u2014 Recompute EF:\n"
        f"  EF = (EDV \u2212 ESV) / EDV = ({_fmt(edv)} \u2212 {_fmt(esv)}) / {_fmt(edv)} = {_fmt(recomputed_ef)}%\n"
        f"  Reported EF: {_fmt(ef)}% | Recomputed: {_fmt(recomputed_ef)}% | \u0394 = {_fmt(ef_diff)}%\n"
    )
    if ef_diff < 1.0:
        answer += "  \u2713 Values are consistent.\n\n"
    else:
        answer += f"  \u26a0 Discrepancy of {_fmt(ef_diff)}% \u2014 minor rounding differences.\n\n"

    answer += "CONSISTENCY CHECK:\n"
    if str(pathology) not in ("nan", "unknown", "None"):
        is_consistent = (
            (ef < 45 and pathology in ("DCM", "MINF", "P"))
            or (ef >= 55 and pathology in ("NOR", "N", "HCM"))
            or (45 <= ef < 55)
        )
        answer += f"  Pathology = {pathology}, EF = {_fmt(ef)}%\n"
        answer += f"  {'\u2713 Diagnosis and EF are concordant.' if is_consistent else '\u26a0 EF and diagnosis may warrant review.'}\n"

    if rv_ef is not None:
        answer += f"  RV EF = {_fmt(rv_ef)}% \u2014 {'normal' if rv_ef >= 45 else 'reduced'}.\n"

    answer += (
        f"RANGE CHECK:\n"
        f"  EDV {_fmt(edv)} mL {'\u2713 plausible' if 30 < edv < 500 else '\u26a0 unusual'}\n"
        f"  ESV {_fmt(esv)} mL {'\u2713 plausible' if 10 < esv < 400 else '\u26a0 unusual'}\n"
        f"</verify>\n\n"
        f"{ML_I}{ML_E}\n"
        f"All checks passed. Assessment is verified."
    )

    return [VQASample(
        id=f"{ds}_{pid}_verify_ef",
        image=row.get(view_col),
        prompt=(
            "Assess the cardiac function from this MRI, then verify your "
            "measurements with a self-check (recompute EF) and consistency "
            "check (compare diagnosis vs. EF range)."
        ),
        answer=answer,
        category="verification",
        dataset=ds,
        meta={"ef": ef, "recomputed_ef": recomputed_ef},
    )]


# ---------------------------------------------------------------------------
# 5. LONGITUDINAL -- prior-scan retrieval + comparison
# ---------------------------------------------------------------------------


def _longitudinal_pairs(rows: list[dict], ds: str, view_col: str, rng: random.Random) -> list[VQASample]:
    """Create synthetic longitudinal pairs from different patients.

    We pair patients with same pathology but different EF to simulate
    'same patient at two time-points' for training the reasoning pattern.
    """
    ef_col = "lv_ef" if "lv_ef" in rows[0] else "ef" if "ef" in rows[0] else None
    edv_col = "lv_edv" if "lv_edv" in rows[0] else "diastole_volume" if "diastole_volume" in rows[0] else None
    esv_col = "lv_esv" if "lv_esv" in rows[0] else "systole_volume" if "systole_volume" in rows[0] else None
    if ef_col is None or edv_col is None or esv_col is None:
        return []

    usable = [r for r in rows if _maybe_float(r.get(ef_col)) is not None and _maybe_float(r.get(edv_col)) is not None]
    if len(usable) < 2:
        return []

    by_path: dict[str, list[dict]] = {}
    for r in usable:
        p = str(r.get("pathology", "unknown"))
        by_path.setdefault(p, []).append(r)

    samples = []
    for pathology, group in by_path.items():
        if len(group) < 2:
            continue
        rng.shuffle(group)
        for i in range(0, len(group) - 1, 2):
            prior, current = group[i], group[i + 1]
            prior_ef = _maybe_float(prior[ef_col])
            curr_ef = _maybe_float(current[ef_col])
            prior_edv = _maybe_float(prior[edv_col])
            curr_edv = _maybe_float(current[edv_col])
            prior_esv = _maybe_float(prior[esv_col])
            curr_esv = _maybe_float(current[esv_col])
            if any(v is None for v in [prior_ef, curr_ef, prior_edv, curr_edv]):
                continue

            delta_ef = curr_ef - prior_ef
            trend = "improved" if delta_ef > 2 else "declined" if delta_ef < -2 else "stable"

            answer = (
                f"<step>RETRIEVE PRIOR</step>\n"
                f"<prior_scan>{prior['pid']}</prior_scan>\n"
                f"Prior study \u2014 EDV: {_fmt(prior_edv)} mL, ESV: {_fmt(prior_esv)} mL, EF: {_fmt(prior_ef)}% ({_ef_severity(prior_ef)})\n"
                f"{MS_I}{MS_E}\n\n"
                f"<step>CURRENT ASSESSMENT</step>\n"
                f"Current study \u2014 EDV: {_fmt(curr_edv)} mL, ESV: {_fmt(curr_esv)} mL, EF: {_fmt(curr_ef)}% ({_ef_severity(curr_ef)})\n\n"
                f"<step>COMPARE</step>\n"
                f"\u0394EF = {'+' if delta_ef >= 0 else ''}{_fmt(delta_ef)} percentage points\n"
                f"\u0394EDV = {'+' if (curr_edv - prior_edv) >= 0 else ''}{_fmt(curr_edv - prior_edv)} mL\n"
                f"Trend: LV systolic function has **{trend}**.\n"
            )
            if trend == "declined":
                answer += "Clinical action: consider therapy escalation or further evaluation.\n"
            elif trend == "improved":
                answer += "Clinical note: positive response; continue current management.\n"
            else:
                answer += "Clinical note: function is stable; continue surveillance.\n"

            answer += f"\n{ML_I}{ML_E}"

            samples.append(VQASample(
                id=f"{ds}_{current['pid']}_longitudinal_vs_{prior['pid']}",
                image=current.get(view_col),
                prompt=(
                    "Compare this cardiac MRI with the patient's prior scan. "
                    "Retrieve the prior measurements, assess the current study, "
                    "and describe any changes in cardiac function over time."
                ),
                answer=answer,
                category="longitudinal",
                dataset=ds,
                meta={
                    "prior_pid": prior["pid"],
                    "prior_ef": prior_ef,
                    "current_ef": curr_ef,
                    "delta_ef": delta_ef,
                    "trend": trend,
                },
            ))
    return samples


# ---------------------------------------------------------------------------
# 6. MULTI-VIEW -- cross-view, cross-phase, wall-motion, RV/LV comparison
# ---------------------------------------------------------------------------


def _multiview_cross_view(rows: list[dict], ds: str) -> list[VQASample]:
    """Cross-view consistency (SAX vs LAX 4CH) for datasets with both."""
    if ds not in ("mnms2", "kaggle"):
        return []

    samples = []
    for row in rows:
        pid = row["pid"]
        sax = row.get("sax_ed") or row.get("sax_t")
        lax = row.get("lax_4c_ed") or row.get("lax_4c_t")
        if not sax or not lax:
            continue

        ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
        edv = _maybe_float(row.get("lv_edv") or row.get("diastole_volume"))
        if ef is None:
            continue

        answer = (
            f"<step>OBSERVE</step>\n"
            f"I examine both the short-axis (SAX) and long-axis 4-chamber (LAX 4CH) views.\n"
            f"SAX view: provides cross-sectional visualization of LV and RV chambers.\n"
            f"LAX 4CH view: shows all four chambers with mitral and tricuspid valve planes.\n"
            f"{MS_I}{MS_E}\n\n"
            f"<step>COMPARE VIEWS</step>\n"
            f"Cross-referencing SAX and LAX views for consistency:\n"
            f"- LV chamber size appears {'dilated' if edv and edv > 200 else 'normal'} in both views.\n"
            f"- Wall motion {'is reduced' if ef < 45 else 'appears preserved'} across views.\n\n"
            f"<step>REASON</step>\n"
            f"LV EF = {_fmt(ef)}% ({_ef_severity(ef)}). "
            f"The cross-view assessment is concordant.\n"
            f"{ML_I}{ML_E}\n\n"
            f"<step>CONCLUDE</step>\n"
            f"Multi-view analysis confirms consistent findings between SAX and LAX 4CH views."
        )

        samples.append(VQASample(
            id=f"{ds}_{pid}_multiview_crossview",
            image=sax,
            prompt=(
                "Compare the short-axis and long-axis 4-chamber views of this cardiac MRI. "
                "Are the findings consistent across views? Describe any discrepancies."
            ),
            answer=answer,
            category="multi_view",
            dataset=ds,
            meta={"views": ["SAX", "LAX_4CH"], "ef": ef},
        ))
    return samples


def _multiview_cross_phase(rows: list[dict], ds: str) -> list[VQASample]:
    """ED vs ES phase comparison."""
    samples = []
    for row in rows:
        pid = row["pid"]
        sax_ed = row.get("sax_ed")
        sax_es = row.get("sax_es")
        if not sax_ed or not sax_es:
            continue

        ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
        edv = _maybe_float(row.get("lv_edv"))
        esv = _maybe_float(row.get("lv_esv"))
        if ef is None or edv is None or esv is None:
            continue

        sv = edv - esv

        answer = (
            f"<step>OBSERVE</step>\n"
            f"Comparing end-diastolic (ED) and end-systolic (ES) phases:\n"
            f"ED: LV cavity is at maximum volume, walls are relaxed.\n"
            f"ES: LV cavity is at minimum volume after contraction.\n"
            f"{MS_I}{MS_E}\n\n"
            f"<step>MEASURE</step>\n"
            f"EDV = {_fmt(edv)} mL (end-diastole)\n"
            f"ESV = {_fmt(esv)} mL (end-systole)\n"
            f"Stroke Volume = {_fmt(sv)} mL\n\n"
            f"<step>REASON</step>\n"
            f"EF = (EDV \u2212 ESV) / EDV = ({_fmt(edv)} \u2212 {_fmt(esv)}) / {_fmt(edv)} = {_fmt(ef)}%\n"
            f"Classification: **{_ef_severity(ef)}**\n"
        )
        if ef < 40:
            answer += "The limited change between ED and ES suggests significant systolic dysfunction.\n"
        elif ef > 60:
            answer += "The marked volume change between phases indicates vigorous systolic function.\n"
        else:
            answer += "The volume change between phases is within normal-to-borderline range.\n"

        answer += (
            f"{ML_I}{ML_E}\n\n"
            f"<step>CONCLUDE</step>\n"
            f"Cross-phase analysis: EF = {_fmt(ef)}% ({_ef_severity(ef)}), SV = {_fmt(sv)} mL."
        )

        samples.append(VQASample(
            id=f"{ds}_{pid}_multiview_crossphase",
            image=sax_ed,
            prompt=(
                "Compare the end-diastolic and end-systolic phases of this cardiac cine-MRI. "
                "Assess the contractile function by analyzing the volume change between phases."
            ),
            answer=answer,
            category="multi_view",
            dataset=ds,
            meta={"ef": ef, "edv": edv, "esv": esv, "sv": sv},
        ))
    return samples


def _multiview_wall_motion(rows: list[dict], ds: str) -> list[VQASample]:
    """Regional wall motion assessment across views."""
    if ds not in ("acdc", "mnms", "mnms2"):
        return []

    samples = []
    for row in rows:
        pid = row["pid"]
        sax_ed = row.get("sax_ed")
        if not sax_ed:
            continue

        ef = _maybe_float(row.get("lv_ef") or row.get("ef"))
        if ef is None:
            continue

        pathology = row.get("pathology", "unknown")
        if str(pathology) in ("nan", "unknown", "None"):
            continue

        regional_desc = {
            "DCM": "global hypokinesis with diffusely reduced wall thickening",
            "HCM": "asymmetric septal hypertrophy with preserved or hyperdynamic basal segments",
            "MINF": "regional akinesis or dyskinesis in the territory of prior infarction",
            "ARV": "RV free wall akinesis with possible aneurysmal dilation",
            "NOR": "normal wall motion in all segments with symmetric thickening",
        }
        desc = regional_desc.get(str(pathology), "wall motion pattern under assessment")

        answer = (
            f"<step>OBSERVE</step>\n"
            f"Assessing regional wall motion across the short-axis stack.\n"
            f"Pattern: {desc}.\n"
            f"{MS_I}{MS_E}\n\n"
            f"<step>REASON</step>\n"
            f"EF = {_fmt(ef)}% ({_ef_severity(ef)}).\n"
            f"The wall motion pattern is consistent with {pathology}.\n"
            f"{ML_I}{ML_E}\n\n"
            f"<step>CONCLUDE</step>\n"
            f"Regional wall motion assessment: {desc}. "
            f"Overall LV systolic function is {_ef_severity(ef)}."
        )

        samples.append(VQASample(
            id=f"{ds}_{pid}_multiview_wallmotion",
            image=sax_ed,
            prompt=(
                "Assess the regional wall motion in this cardiac cine-MRI. "
                "Describe the pattern of wall thickening across segments and "
                "correlate with overall ventricular function."
            ),
            answer=answer,
            category="multi_view",
            dataset=ds,
            meta={"ef": ef, "pathology": str(pathology)},
        ))
    return samples


def _multiview_rv_lv_compare(rows: list[dict], ds: str) -> list[VQASample]:
    """Biventricular comparison (RV vs LV function)."""
    samples = []
    for row in rows:
        pid = row["pid"]
        sax_ed = row.get("sax_ed") or row.get("sax_t")
        if not sax_ed:
            continue

        lv_ef = _maybe_float(row.get("lv_ef"))
        rv_edv = _maybe_float(row.get("rv_edv"))
        rv_esv = _maybe_float(row.get("rv_esv"))
        lv_edv = _maybe_float(row.get("lv_edv"))
        lv_esv = _maybe_float(row.get("lv_esv"))

        if lv_ef is None or rv_edv is None or rv_esv is None:
            continue
        if lv_edv is None or lv_esv is None:
            continue
        if rv_edv <= 0:
            continue

        rv_ef = (rv_edv - rv_esv) / rv_edv * 100

        answer = (
            f"<step>OBSERVE</step>\n"
            f"Examining both left and right ventricles in the short-axis view.\n"
            f"{MS_I}{MS_E}\n\n"
            f"<step>MEASURE</step>\n"
            f"Left Ventricle:  EDV = {_fmt(lv_edv)} mL, ESV = {_fmt(lv_esv)} mL, EF = {_fmt(lv_ef)}%\n"
            f"Right Ventricle: EDV = {_fmt(rv_edv)} mL, ESV = {_fmt(rv_esv)} mL, EF = {_fmt(rv_ef)}%\n\n"
            f"<step>REASON</step>\n"
            f"LV function: {_ef_severity(lv_ef)} (EF {_fmt(lv_ef)}%)\n"
            f"RV function: {'normal' if rv_ef >= 45 else 'reduced'} (EF {_fmt(rv_ef)}%)\n"
        )

        if lv_ef < 45 and rv_ef < 45:
            answer += "Both ventricles show reduced function \u2014 consider biventricular failure.\n"
        elif lv_ef < 45:
            answer += "Isolated LV dysfunction with preserved RV function.\n"
        elif rv_ef < 45:
            answer += "Isolated RV dysfunction with preserved LV function \u2014 consider pulmonary hypertension or RV pathology.\n"
        else:
            answer += "Both ventricles demonstrate preserved systolic function.\n"

        answer += (
            f"{ML_I}{ML_E}\n\n"
            f"<step>CONCLUDE</step>\n"
            f"Biventricular assessment: LV EF = {_fmt(lv_ef)}%, RV EF = {_fmt(rv_ef)}%."
        )

        samples.append(VQASample(
            id=f"{ds}_{pid}_multiview_rv_lv",
            image=sax_ed,
            prompt=(
                "Compare the left and right ventricular function in this cardiac MRI. "
                "Measure both LV and RV volumes and ejection fractions, and assess "
                "whether there is isolated or biventricular dysfunction."
            ),
            answer=answer,
            category="multi_view",
            dataset=ds,
            meta={"lv_ef": lv_ef, "rv_ef": rv_ef},
        ))
    return samples


# ---------------------------------------------------------------------------
# Dataset-specific generators
# ---------------------------------------------------------------------------


def _primary_image_col(ds: str) -> str:
    """Return the CSV column holding the primary image path for a dataset."""
    if ds in ("acdc", "mnms", "mnms2"):
        return "sax_ed"
    if ds == "kaggle":
        return "sax_t"
    return "image"


def generate_for_dataframe(df: pd.DataFrame, ds: str, rng: random.Random) -> list[VQASample]:
    """Generate all VQA categories for a single dataset DataFrame."""
    rows = df.to_dict(orient="records")
    view_col = _primary_image_col(ds)
    samples: list[VQASample] = []

    for row in rows:
        samples.extend(_basic_ef(row, ds, view_col))
        samples.extend(_basic_pathology(row, ds, view_col))
        samples.extend(_basic_volume(row, ds, view_col))
        samples.extend(_multistep_ef_analysis(row, ds, view_col))
        samples.extend(_multistep_morphology(row, ds, view_col))
        samples.extend(_tool_segment_and_measure(row, ds, view_col))
        samples.extend(_verify_ef(row, ds, view_col))

    samples.extend(_longitudinal_pairs(rows, ds, view_col, rng))
    samples.extend(_multiview_cross_view(rows, ds))
    samples.extend(_multiview_cross_phase(rows, ds))
    samples.extend(_multiview_wall_motion(rows, ds))
    samples.extend(_multiview_rv_lv_compare(rows, ds))
    return samples


def build_all(
    data_dir: Path,
    out_dir: Path,
    datasets: list[str],
    seed: int = 42,
) -> None:
    """Build memory-annotated VQA JSONL for all datasets and splits."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        all_samples: list[VQASample] = []

        for ds in datasets:
            csv_path = data_dir / ds / f"{split}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, dtype={"pid": str})
            samples = generate_for_dataframe(df, ds, rng)
            all_samples.extend(samples)

        if not all_samples:
            continue

        rng.shuffle(all_samples)
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(s.to_jsonl() + "\n")


# ---------------------------------------------------------------------------
# Ground-truth annotation for CineMem training
# ---------------------------------------------------------------------------

# Categories where cross-view comparisons occur
_CROSS_VIEW_CATEGORIES = {
    "biventricular_comparison",
    "ed_es_phase_comparison",
    "cross_view_structure",
    "multi_view_contraction",
    "cross_view_consistency_verification",
}

# Categories where a diagnostic conclusion is drawn
_DIAGNOSTIC_CATEGORIES = {
    "abnormality_localization",
    "diagnosis",
    "lv_systolic_function",
    "ventricular_dilation_lv",
    "ventricular_dilation_rv",
}


def annotate_ground_truth(gt_entry: dict) -> str:
    """Wrap a CineMA ground-truth entry with think/answer tags and memory tokens.

    Inserts ``<ms_I><ms_E>`` before cross-view comparisons and quantitative
    claims.  Inserts ``<ml_I><ml_E>`` before diagnostic conclusions.

    Args:
        gt_entry: a single row dict from CineMA's ``ground_truth_*.jsonl``.

    Returns:
        A string formatted as ``<think>...</think>\\n<answer>(...)</answer>``
        suitable for CineMem-style supervised training.
    """
    reasoning = gt_entry.get("reasoning_template", "")
    answer_letter = gt_entry.get("correct_option", "")
    category = gt_entry.get("category", "")

    parts: list[str] = []

    if category in _CROSS_VIEW_CATEGORIES:
        parts.append(f"{MS_I}{MS_E}")

    if reasoning:
        parts.append(reasoning)

    if category in _DIAGNOSTIC_CATEGORIES:
        parts.append(f"{ML_I}{ML_E}")

    think_block = "\n".join(parts) if parts else reasoning
    return f"<think>{think_block}</think>\n<answer>({answer_letter})</answer>"
