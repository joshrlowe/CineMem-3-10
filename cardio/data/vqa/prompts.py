"""System prompts and per-category reasoning templates for GPT-4o VQA generation."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt — used for every GPT-4o reasoning call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_REASONING = """\
You are a cardiac MRI interpretation assistant. You will be given a patient's \
cardiac measurements, a multiple-choice clinical question, the correct answer, \
and a bounding box identifying the relevant anatomical structure.

Your task is to write a concise clinical reasoning explanation that:
1. References the patient's EXACT measurements (do not round or alter values).
2. Explains why the correct answer is right.
3. Briefly explains why the other options are ruled out.
4. Uses varied, natural clinical language (do not copy the template verbatim).

Output your response as a JSON object with exactly these fields:
{
  "correct_option": "<letter>",
  "reasoning": "<your clinical reasoning text>",
  "grounding": {
    "structure": "<structure name>",
    "bbox": [x1, y1, x2, y2],
    "source_mask_label": <integer label>
  }
}

IMPORTANT:
- Do NOT cite mL/m2 thresholds when comparing against absolute mL values.
- Use qualitative volume language (e.g., "substantially exceeds expected ranges").
- Keep reasoning to 2-4 sentences.
- The correct_option, grounding structure, bbox, and source_mask_label must \
match the values provided to you exactly.
"""

# ---------------------------------------------------------------------------
# System prompt — used for question paraphrasing calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PARAPHRASE = """\
You are a medical education content writer. You will be given a multiple-choice \
clinical question about cardiac MRI. Generate exactly 3 alternative phrasings of \
the question that ask the same clinical question but use different wording.

Rules:
1. Keep the EXACT same answer options and option letters (A, B, C, etc.).
2. Each paraphrase must be clinically appropriate.
3. Do not change the meaning of the question.
4. Output as a JSON array of 3 strings, each being a complete rephrased question \
(without the options — only the question stem).

Example output:
["Rephrased question 1", "Rephrased question 2", "Rephrased question 3"]
"""

# ---------------------------------------------------------------------------
# Per-category reasoning templates
#
# These are provided to GPT-4o as a *guide* for style and content.
# The LLM is instructed to vary the wording.
# Placeholders: {lv_edv}, {lv_esv}, {lv_ef}, {rv_edv}, {rv_esv}, {rv_ef}
# ---------------------------------------------------------------------------

CAT1_TEMPLATES: dict[str, str] = {
    "DCM": (
        "The left ventricular cavity at the indicated region is significantly dilated. "
        "The LV end-diastolic volume is {lv_edv} mL, which is substantially elevated "
        "above expected normal ranges. The ejection fraction is {lv_ef}% (severely "
        "reduced, normal >55%). These findings are consistent with left ventricular "
        "dilation and impaired systolic function. The myocardium and right ventricle "
        "appear within normal limits."
    ),
    "DLV": (
        "The left ventricular cavity at the indicated region is significantly dilated. "
        "The LV end-diastolic volume is {lv_edv} mL, which is substantially elevated "
        "above expected normal ranges. The ejection fraction is {lv_ef}% (severely "
        "reduced, normal >55%). These findings are consistent with left ventricular "
        "dilation and impaired systolic function. The myocardium and right ventricle "
        "appear within normal limits."
    ),
    "LV": (
        "The left ventricular cavity at the indicated region is significantly dilated. "
        "The LV end-diastolic volume is {lv_edv} mL, which is substantially elevated "
        "above expected normal ranges. The ejection fraction is {lv_ef}% (severely "
        "reduced, normal >55%). These findings are consistent with left ventricular "
        "dilation and impaired systolic function. The myocardium and right ventricle "
        "appear within normal limits."
    ),
    "HCM": (
        "The myocardium at the indicated region appears thickened, consistent with "
        "the HCM pathology label. The ejection fraction is preserved at {lv_ef}% "
        "(normal >55%), which is characteristic of hypertrophic cardiomyopathy where "
        "the heart muscle thickens but pump function is maintained. The ventricular "
        "cavities are not dilated."
    ),
    "ARV": (
        "The right ventricle cavity at the indicated region appears significantly "
        "dilated. The RV end-diastolic volume is {rv_edv} mL, which substantially "
        "exceeds expected normal ranges. Combined with a reduced RV ejection fraction "
        "of {rv_ef}%, these findings indicate right ventricular dilation and "
        "dysfunction. The left ventricle appears normal in size."
    ),
    "RV": (
        "The right ventricle cavity at the indicated region appears significantly "
        "dilated. The RV end-diastolic volume is {rv_edv} mL, which substantially "
        "exceeds expected normal ranges. Combined with a reduced RV ejection fraction "
        "of {rv_ef}%, these findings indicate right ventricular dilation and "
        "dysfunction. The left ventricle appears normal in size."
    ),
    "ARR": (
        "The right ventricle cavity at the indicated region appears significantly "
        "dilated. The RV end-diastolic volume is {rv_edv} mL, which substantially "
        "exceeds expected normal ranges. Combined with a reduced RV ejection fraction "
        "of {rv_ef}%, these findings indicate right ventricular dilation and "
        "dysfunction. The left ventricle appears normal in size."
    ),
    "CIA": (
        "The right ventricle cavity at the indicated region appears significantly "
        "dilated. The RV end-diastolic volume is {rv_edv} mL, which substantially "
        "exceeds expected normal ranges. Combined with a reduced RV ejection fraction "
        "of {rv_ef}%, these findings indicate right ventricular dilation and "
        "dysfunction. The left ventricle appears normal in size."
    ),
    "FALL": (
        "The right ventricle cavity at the indicated region appears significantly "
        "dilated. The RV end-diastolic volume is {rv_edv} mL, which substantially "
        "exceeds expected normal ranges. Combined with a reduced RV ejection fraction "
        "of {rv_ef}%, these findings indicate right ventricular dilation and "
        "dysfunction. The left ventricle appears normal in size."
    ),
    "MINF": (
        "The left ventricle at the indicated region does not appear significantly "
        "dilated, but shows evidence of reduced systolic function. The ejection "
        "fraction is {lv_ef}% (reduced, normal >55%), indicating impaired "
        "contractility. Unlike dilated cardiomyopathy, the LV cavity size is not "
        "markedly enlarged. The myocardium and right ventricle appear structurally "
        "normal."
    ),
    "NOR": (
        "All cardiac structures appear within normal limits. The LV ejection "
        "fraction is {lv_ef}% (normal >55%), the LV end-diastolic volume is within "
        "expected normal ranges, and the RV function is preserved (RV EF {rv_ef}%, "
        "normal >40%). No structural abnormality is identified."
    ),
}

CAT2_TEMPLATES: dict[str, str] = {
    "A": (
        "The left ventricle at the indicated region demonstrates normal systolic "
        "function. The ejection fraction is {lv_ef}%, which exceeds the 55% "
        "threshold for normal function. The LV cavity shows appropriate reduction "
        "in size from end-diastole to end-systole, indicating effective contraction."
    ),
    "B": (
        "The left ventricle at the indicated region demonstrates mildly reduced "
        "systolic function. The ejection fraction is {lv_ef}%, falling between the "
        "41% and 55% thresholds. While not severely impaired, this represents a "
        "departure from normal function and warrants clinical follow-up."
    ),
    "C": (
        "The left ventricle at the indicated region demonstrates moderately reduced "
        "systolic function. The ejection fraction is {lv_ef}%, falling between the "
        "30% and 40% thresholds. This indicates meaningful impairment of contractile "
        "function with clinical implications for management."
    ),
    "D": (
        "The left ventricle at the indicated region demonstrates severely reduced "
        "systolic function. The ejection fraction is {lv_ef}%, well below the 30% "
        "threshold for severe impairment. The LV cavity shows minimal change in "
        "size from end-diastole to end-systole, indicating very poor contractility."
    ),
}

CAT3_LV_TEMPLATES: dict[str, str] = {
    "A": (
        "The left ventricle at the indicated region appears significantly dilated. "
        "The LV end-diastolic volume is {lv_edv} mL, which substantially exceeds "
        "normal limits. Combined with a reduced ejection fraction of {lv_ef}%, "
        "these findings are consistent with left ventricular dilation."
    ),
    "B": (
        "The left ventricle at the indicated region appears normal in size, with an "
        "LV end-diastolic volume of {lv_edv} mL and an ejection fraction of "
        "{lv_ef}%. No ventricular dilation is identified."
    ),
}

CAT3_RV_TEMPLATES: dict[str, str] = {
    "A": (
        "The right ventricle at the indicated region appears significantly dilated. "
        "The RV end-diastolic volume is {rv_edv} mL, which substantially exceeds "
        "expected normal ranges. Combined with a reduced RV ejection fraction of "
        "{rv_ef}%, these findings indicate right ventricular dilation."
    ),
    "B": (
        "The right ventricle at the indicated region appears normal in size (RV EDV "
        "{rv_edv} mL, RV EF {rv_ef}%). No right ventricular dilation is identified."
    ),
}

CAT4_TEMPLATES: dict[str, str] = {
    "DCM": (
        "The left ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {lv_edv} mL, substantially "
        "exceeding expected normal ranges. The ejection fraction is severely "
        "reduced at {lv_ef}% (criterion: <40%). These findings -- ventricular "
        "dilation combined with reduced systolic function -- are the hallmark "
        "of dilated cardiomyopathy (DCM)."
    ),
    "DLV": (
        "The left ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {lv_edv} mL, substantially "
        "exceeding expected normal ranges. The ejection fraction is reduced at "
        "{lv_ef}%. These findings are consistent with dilated left ventricle (DLV)."
    ),
    "LV": (
        "The left ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {lv_edv} mL, substantially "
        "exceeding expected normal ranges. The ejection fraction is reduced at "
        "{lv_ef}%. These findings are consistent with dilated left ventricle (DLV)."
    ),
    "HCM": (
        "The myocardium at the indicated region appears thickened, consistent "
        "with the HCM pathology label. The ejection fraction is preserved at "
        "{lv_ef}% (criterion: >55%), which distinguishes HCM from other "
        "cardiomyopathies where EF is typically reduced. The ventricular "
        "cavities are not dilated."
    ),
    "ARV": (
        "The right ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {rv_edv} mL, substantially "
        "exceeding expected normal ranges. The RV ejection fraction is reduced "
        "at {rv_ef}% (criterion: <40%). The left ventricle appears normal."
    ),
    "RV": (
        "The right ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {rv_edv} mL, substantially "
        "exceeding expected normal ranges. The RV ejection fraction is reduced "
        "at {rv_ef}% (criterion: <40%). The left ventricle appears normal."
    ),
    "ARR": (
        "The right ventricular cavity at the indicated region is significantly "
        "dilated with an end-diastolic volume of {rv_edv} mL. The RV ejection "
        "fraction is reduced at {rv_ef}%. These findings are consistent with "
        "arrhythmogenic cardiomyopathy (ARR)."
    ),
    "FALL": (
        "The right ventricular cavity at the indicated region appears dilated "
        "with an end-diastolic volume of {rv_edv} mL. The RV ejection fraction "
        "is {rv_ef}%. In this post-surgical repair adult, RV dilation from "
        "chronic pulmonary regurgitation is consistent with Tetralogy of Fallot."
    ),
    "CIA": (
        "The right ventricular cavity at the indicated region appears dilated "
        "with an end-diastolic volume of {rv_edv} mL, indicating RV volume "
        "overload. The RV ejection fraction is {rv_ef}%. These findings are "
        "consistent with inter-atrial communication (CIA)."
    ),
    "MINF": (
        "The left ventricular cavity at the indicated region shows reduced "
        "systolic function with an ejection fraction of {lv_ef}% (criterion: "
        "<40%). While the LV is not significantly dilated (distinguishing this "
        "from DCM), the reduced EF indicates myocardial infarction (MINF)."
    ),
    "NOR": (
        "All cardiac structures appear within normal limits. The LV ejection "
        "fraction is {lv_ef}% (normal >55%), the ventricular volumes are "
        "within expected ranges, and the RV function is preserved at {rv_ef}%. "
        "The findings are consistent with a normal cardiac study."
    ),
}

CAT5_TEMPLATES: dict[str, str] = {
    "A": (
        "The left ventricle at the indicated region is significantly dilated "
        "(EDV {lv_edv} mL) with reduced systolic function (EF {lv_ef}%). In "
        "contrast, the right ventricle appears normal in size (EDV {rv_edv} mL) "
        "with preserved function (EF {rv_ef}%). The pathology predominantly "
        "affects the left ventricle, sparing the right ventricle."
    ),
    "B": (
        "The right ventricle at the indicated region is significantly dilated "
        "(EDV {rv_edv} mL) with reduced function (EF {rv_ef}%). The left "
        "ventricle appears normal in size (EDV {lv_edv} mL) with preserved "
        "function (EF {lv_ef}%). The pathology predominantly affects the right "
        "ventricle, sparing the left ventricle."
    ),
    "D": (
        "Both ventricles appear normal. The left ventricle (EDV {lv_edv} mL, "
        "EF {lv_ef}%) and the right ventricle (EDV {rv_edv} mL, EF {rv_ef}%) "
        "demonstrate normal size and function. No asymmetric findings are "
        "identified."
    ),
    "E": (
        "The left ventricle at the indicated region is not significantly "
        "dilated (EDV {lv_edv} mL) but demonstrates reduced systolic function "
        "(EF {lv_ef}%). The right ventricle appears normal in both size "
        "(EDV {rv_edv} mL) and function (EF {rv_ef}%). Unlike dilated "
        "cardiomyopathy, the primary finding is impaired contractility "
        "without significant dilation."
    ),
}

CAT6_TEMPLATES: dict[str, str] = {
    "A": (
        "Comparing the ED and ES frames, the left ventricle at the indicated "
        "region demonstrates significant reduction in cavity size. The LV volume "
        "decreases from {lv_edv} mL at end-diastole to {lv_esv} mL at "
        "end-systole, corresponding to an ejection fraction of {lv_ef}%. This "
        "represents normal contractile function."
    ),
    "B": (
        "Comparing the ED and ES frames, the left ventricle at the indicated "
        "region shows moderate reduction in cavity size. The LV volume decreases "
        "from {lv_edv} mL at end-diastole to {lv_esv} mL at end-systole, "
        "corresponding to an ejection fraction of {lv_ef}%. Contraction is "
        "present but mildly impaired compared to normal."
    ),
    "C": (
        "Comparing the ED and ES frames, the left ventricle at the indicated "
        "region shows minimal reduction in cavity size. The LV volume decreases "
        "only from {lv_edv} mL at end-diastole to {lv_esv} mL at end-systole, "
        "corresponding to an ejection fraction of only {lv_ef}%. The heart is "
        "pumping less than 40% of its blood volume per beat, indicating "
        "significantly impaired contractility."
    ),
}


def build_reasoning_prompt(
    *,
    category: str,
    question_text: str,
    options: list[dict[str, str]],
    correct_option: str,
    metrics: dict[str, float],
    grounding: dict,
    reasoning_template: str,
) -> str:
    """Build the user-message prompt for GPT-4o reasoning generation.

    Args:
        category: category identifier string.
        question_text: the question stem.
        options: list of {"id": ..., "text": ...} option dicts.
        correct_option: letter of the correct answer.
        metrics: dict of patient metrics (lv_edv, lv_ef, etc.).
        grounding: dict with structure, bbox, source_mask_label.
        reasoning_template: the template string to guide the LLM.

    Returns:
        Formatted user-message string.
    """
    options_str = "\n".join(f"({o['id']}) {o['text']}" for o in options)
    metrics_str = "\n".join(f"- {k}: {v}" for k, v in metrics.items())

    if "structures" in grounding:
        grounding_lines = []
        for s in grounding["structures"]:
            grounding_lines.append(
                f"  - structure={s['structure']}, bbox={s['bbox']}, "
                f"source_mask_label={s['source_mask_label']}"
            )
        grounding_str = "Grounding (multiple structures):\n" + "\n".join(grounding_lines)
    else:
        grounding_str = (
            f"Grounding: structure={grounding.get('structure', 'NONE')}, "
            f"bbox={grounding.get('bbox', [])}, "
            f"source_mask_label={grounding.get('source_mask_label', -1)}"
        )

    return f"""\
Patient metrics:
{metrics_str}

Question ({category}):
{question_text}
{options_str}

Correct answer: ({correct_option})
{grounding_str}

Reasoning template (use as a GUIDE for content, but vary the wording):
{reasoning_template}
"""


def build_paraphrase_prompt(
    *,
    question_text: str,
    options: list[dict[str, str]],
) -> str:
    """Build the user-message prompt for question paraphrasing.

    Args:
        question_text: the original question stem.
        options: list of {"id": ..., "text": ...} option dicts.

    Returns:
        Formatted user-message string.
    """
    options_str = "\n".join(f"({o['id']}) {o['text']}" for o in options)

    return f"""\
Original question:
{question_text}

Options (keep these EXACTLY as-is):
{options_str}

Generate 3 alternative phrasings of the question stem only. \
Output as a JSON array of 3 strings.
"""
