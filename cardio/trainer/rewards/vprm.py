"""Deterministic Verifiable Process Reward Model for cardiac VQA.

Implements ACC/AHA 2022 guideline-based verification of VLM reasoning.
Extracts quantitative cardiac metrics from generated ``<think>`` blocks
using regex patterns and applies clinical logic trees to verify reasoning
consistency, producing a scalar reward signal for GRPO training.
"""

from __future__ import annotations

import re

from cardio.data.vqa import VQACategory

# ---------------------------------------------------------------------------
# Compiled regex patterns for metric extraction
# ---------------------------------------------------------------------------

_LVEF_RE = re.compile(
    r"LVEF\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)
_LAVI_RE = re.compile(
    r"LAVI\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*mL/m", re.IGNORECASE
)
_GLS_RE = re.compile(
    r"GLS\s*(?:is|of|=|:)?\s*(-?\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)
_E_E_PRIME_RE = re.compile(
    r"E/e['\u2032]?\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_LVMI_RE = re.compile(
    r"LVMI\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*g/m", re.IGNORECASE
)
_RWT_RE = re.compile(
    r"RWT\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)

_METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "lvef": _LVEF_RE,
    "lavi": _LAVI_RE,
    "gls": _GLS_RE,
    "e_e_prime": _E_E_PRIME_RE,
    "lvmi": _LVMI_RE,
    "rwt": _RWT_RE,
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# HF subtype normalisation helpers
# ---------------------------------------------------------------------------

_HF_KEYWORD_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"HFrEF", re.IGNORECASE), "HFrEF"),
    (re.compile(r"HFmrEF", re.IGNORECASE), "HFmrEF"),
    (re.compile(r"HFpEF", re.IGNORECASE), "HFpEF"),
    (re.compile(r"HFimpEF", re.IGNORECASE), "HFimpEF"),
    (re.compile(r"reduced\s+ejection\s+fraction", re.IGNORECASE), "HFrEF"),
    (re.compile(r"mildly\s+reduced\s+ejection\s+fraction", re.IGNORECASE), "HFmrEF"),
    (re.compile(r"preserved\s+ejection\s+fraction", re.IGNORECASE), "HFpEF"),
    (re.compile(r"improved\s+ejection\s+fraction", re.IGNORECASE), "HFimpEF"),
]

# ---------------------------------------------------------------------------
# Contraction descriptors for EF-consistency checks
# ---------------------------------------------------------------------------

_CONTRACTION_DESCRIPTORS: list[tuple[re.Pattern[str], tuple[float, float]]] = [
    (re.compile(r"severely\s+reduced\s+contraction", re.IGNORECASE), (0.0, 30.0)),
    (re.compile(r"significantly\s+impaired\s+contraction", re.IGNORECASE), (0.0, 30.0)),
    (re.compile(r"moderately\s+reduced\s+contraction", re.IGNORECASE), (20.0, 45.0)),
    (re.compile(r"mildly\s+(?:reduced|impaired)\s+contraction", re.IGNORECASE), (35.0, 55.0)),
    (re.compile(r"normal\s+contraction", re.IGNORECASE), (50.0, 100.0)),
    (re.compile(r"hyperdynamic\s+contraction", re.IGNORECASE), (60.0, 100.0)),
    (re.compile(r"severely\s+reduced\s+(?:systolic\s+)?function", re.IGNORECASE), (0.0, 30.0)),
    (re.compile(r"moderately\s+reduced\s+(?:systolic\s+)?function", re.IGNORECASE), (20.0, 45.0)),
    (re.compile(r"mildly\s+reduced\s+(?:systolic\s+)?function", re.IGNORECASE), (35.0, 55.0)),
    (re.compile(r"normal\s+(?:systolic\s+)?function", re.IGNORECASE), (50.0, 100.0)),
]

# AHA 17-segment model keywords for wall motion assessment
_SEGMENT_KEYWORDS = re.compile(
    r"\b(?:basal|mid|apical|apex|anterior|inferior|septal|lateral|"
    r"anteroseptal|anterolateral|inferoseptal|inferolateral)\b",
    re.IGNORECASE,
)

_WALL_MOTION_ABNORMALITY_RE = re.compile(
    r"(?:wall\s+motion\s+abnormalit|akinesi[as]|hypokinesi[as]|dyskinesi[as]|akinetic|hypokinetic|dyskinetic)",
    re.IGNORECASE,
)

# Categories routed to HF classification verification
_HF_CATEGORIES: set[str] = {
    VQACategory.DIAGNOSIS.value,
}


def _extract_think_block(text: str) -> str:
    """Return content inside ``<think>`` tags, or the full text as fallback."""
    m = _THINK_RE.search(text)
    return m.group(1) if m else text


def _normalise_hf_label(text: str) -> str | None:
    """Map free-text to a canonical HF subtype, or ``None`` if unrecognised.

    Longer patterns (e.g. "mildly reduced ejection fraction") are checked
    before shorter ones ("reduced ejection fraction") so the ordering of
    ``_HF_KEYWORD_MAP`` matters.
    """
    for pattern, label in _HF_KEYWORD_MAP:
        if pattern.search(text):
            return label
    return None


# =========================================================================
# Main verifier
# =========================================================================


class ACCAHAVerifier:
    """Deterministic Verifiable Process Reward Model for cardiac VQA.

    Extracts quantitative metrics from generated text using regex patterns
    and applies ACC/AHA 2022 logic trees to verify reasoning consistency.
    """

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def extract_metrics(self, text: str) -> dict[str, float | None]:
        """Extract all quantitative metrics from the reasoning text."""
        metrics: dict[str, float | None] = {}
        for name, pattern in _METRIC_PATTERNS.items():
            m = pattern.search(text)
            metrics[name] = float(m.group(1)) if m else None
        return metrics

    # ------------------------------------------------------------------
    # HF classification (ACC/AHA 2022)
    # ------------------------------------------------------------------

    def verify_hf_classification(
        self,
        text: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> tuple[float, list[str]]:
        """Verify Heart Failure classification against ACC/AHA 2022 thresholds.

        Returns ``(reward, violations)`` where reward is one of
        ``{-1.0, 0.0, 0.5, 1.0}``.
        """
        violations: list[str] = []

        pred_label = _normalise_hf_label(predicted_answer)
        gt_label = _normalise_hf_label(ground_truth)

        if pred_label is None:
            violations.append(
                f"Could not normalise predicted answer to HF subtype: {predicted_answer!r}"
            )
            return -1.0, violations

        if gt_label is not None and pred_label != gt_label:
            violations.append(
                f"Incorrect diagnosis: predicted {pred_label}, expected {gt_label}"
            )
            return -1.0, violations

        metrics = self.extract_metrics(text)
        lvef = metrics.get("lvef")

        if pred_label == "HFrEF":
            return self._verify_hfref(metrics, lvef, text, violations)
        if pred_label == "HFmrEF":
            return self._verify_hfmref(metrics, lvef, text, violations)
        if pred_label == "HFpEF":
            return self._verify_hfpef(metrics, lvef, text, violations)
        if pred_label == "HFimpEF":
            return self._verify_hfimpef(lvef, text, violations)

        violations.append(f"Unhandled HF subtype: {pred_label}")
        return -1.0, violations

    # -- HFrEF --------------------------------------------------------

    @staticmethod
    def _verify_hfref(
        metrics: dict[str, float | None],
        lvef: float | None,
        text: str,
        violations: list[str],
    ) -> tuple[float, list[str]]:
        if lvef is None:
            violations.append("HFrEF claimed but LVEF not stated")
            return 0.5, violations
        if lvef > 40:
            violations.append(
                f"HFrEF requires LVEF <= 40%, extracted LVEF = {lvef}%"
            )
            return 0.0, violations
        return 1.0, violations

    # -- HFmrEF -------------------------------------------------------

    @staticmethod
    def _verify_hfmref(
        metrics: dict[str, float | None],
        lvef: float | None,
        text: str,
        violations: list[str],
    ) -> tuple[float, list[str]]:
        if lvef is None:
            violations.append("HFmrEF claimed but LVEF not stated")
            return 0.5, violations
        if not (41 <= lvef <= 49):
            violations.append(
                f"HFmrEF requires 41% <= LVEF <= 49%, extracted LVEF = {lvef}%"
            )
            return 0.0, violations

        filling_evidence = _has_filling_pressure_evidence(metrics, text)
        if not filling_evidence:
            violations.append(
                "HFmrEF requires filling pressure evidence "
                "(LAVI >= 34, E/e' >= 15, or abnormal GLS) — none found"
            )
            return 0.5, violations
        return 1.0, violations

    # -- HFpEF --------------------------------------------------------

    @staticmethod
    def _verify_hfpef(
        metrics: dict[str, float | None],
        lvef: float | None,
        text: str,
        violations: list[str],
    ) -> tuple[float, list[str]]:
        if lvef is None:
            violations.append("HFpEF claimed but LVEF not stated")
            return 0.5, violations
        if lvef < 50:
            violations.append(
                f"HFpEF requires LVEF >= 50%, extracted LVEF = {lvef}%"
            )
            return 0.0, violations

        has_structural = _has_structural_evidence(metrics, text)
        has_filling = _has_filling_pressure_evidence(metrics, text)
        if not (has_structural or has_filling):
            violations.append(
                "HFpEF requires structural or filling pressure evidence — none found"
            )
            return 0.5, violations
        return 1.0, violations

    # -- HFimpEF -------------------------------------------------------

    @staticmethod
    def _verify_hfimpef(
        lvef: float | None,
        text: str,
        violations: list[str],
    ) -> tuple[float, list[str]]:
        prior_ef_re = re.compile(
            r"(?:prior|previous|baseline|formerly)\s+.*?"
            r"LVEF\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        )
        prior_match = prior_ef_re.search(text)
        if prior_match is None:
            violations.append(
                "HFimpEF requires reference to prior LVEF <= 40% — not found"
            )
            return 0.5, violations

        prior_ef = float(prior_match.group(1))
        if prior_ef > 40:
            violations.append(
                f"HFimpEF prior LVEF should be <= 40%, found {prior_ef}%"
            )
            return 0.0, violations

        if lvef is None:
            violations.append("HFimpEF claimed but current LVEF not stated")
            return 0.5, violations
        if lvef <= 40:
            violations.append(
                f"HFimpEF current LVEF should be > 40%, extracted {lvef}%"
            )
            return 0.0, violations

        return 1.0, violations

    # ------------------------------------------------------------------
    # EF consistency (Category 11 / LV function)
    # ------------------------------------------------------------------

    def verify_ef_consistency(self, text: str) -> tuple[float, list[str]]:
        """Check that stated EF aligns with the visual contraction description.

        Used for *cross_view_consistency_verification* (Cat 11) and
        *lv_systolic_function* (Cat 2).
        """
        violations: list[str] = []
        metrics = self.extract_metrics(text)
        lvef = metrics.get("lvef")

        if lvef is None:
            return 1.0, violations

        for descriptor_re, (lo, hi) in _CONTRACTION_DESCRIPTORS:
            if descriptor_re.search(text):
                if not (lo <= lvef <= hi):
                    violations.append(
                        f"EF-contraction mismatch: LVEF {lvef}% inconsistent "
                        f"with '{descriptor_re.pattern}' (expected {lo}–{hi}%)"
                    )
                    return 0.0, violations

        return 1.0, violations

    # ------------------------------------------------------------------
    # Contraction assessment (Category 8)
    # ------------------------------------------------------------------

    def verify_contraction_assessment(self, text: str) -> tuple[float, list[str]]:
        """Verify wall motion abnormality claims for multi-view contraction.

        Checks that:
        1. Wall motion abnormality claims reference specific AHA segments.
        2. Abnormality severity is consistent with stated EF range.
        """
        violations: list[str] = []

        has_wma = bool(_WALL_MOTION_ABNORMALITY_RE.search(text))

        if has_wma:
            segments_found = _SEGMENT_KEYWORDS.findall(text)
            if not segments_found:
                violations.append(
                    "Wall motion abnormality claimed without naming specific "
                    "AHA segments (e.g. basal, mid, apical, anterior, etc.)"
                )

        metrics = self.extract_metrics(text)
        lvef = metrics.get("lvef")

        if lvef is not None and has_wma:
            if lvef >= 55:
                violations.append(
                    f"Wall motion abnormality claimed with normal LVEF ({lvef}%) "
                    "— unusual without regional specificity"
                )

        if lvef is not None and not has_wma:
            if lvef < 40:
                violations.append(
                    f"LVEF {lvef}% suggests significant dysfunction but no "
                    "wall motion abnormality is described"
                )

        ef_consistency_reward, ef_violations = self.verify_ef_consistency(text)
        violations.extend(ef_violations)

        if violations:
            return 0.0, violations
        return 1.0, violations

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        text: str,
        predicted_answer: str,
        ground_truth: str,
        category: str,
    ) -> tuple[float, dict]:
        """Route to the appropriate verification method based on VQA category.

        Returns ``(reward, metadata_dict)``.
        """
        reasoning = _extract_think_block(text)
        metrics = self.extract_metrics(reasoning)

        metadata: dict = {
            "category": category,
            "metrics": metrics,
            "violations": [],
            "sub_method": "default",
        }

        if category in _HF_CATEGORIES:
            reward, violations = self.verify_hf_classification(
                reasoning, predicted_answer, ground_truth
            )
            metadata["sub_method"] = "hf_classification"
            metadata["violations"] = violations
            return reward, metadata

        if category == VQACategory.CROSS_VIEW_CONSISTENCY.value:
            reward, violations = self.verify_ef_consistency(reasoning)
            metadata["sub_method"] = "ef_consistency"
            metadata["violations"] = violations
            if not violations:
                if predicted_answer.strip().lower() != ground_truth.strip().lower():
                    reward = -1.0
                    violations.append(
                        f"Incorrect answer: '{predicted_answer}' vs '{ground_truth}'"
                    )
                    metadata["violations"] = violations
            return reward, metadata

        if category == VQACategory.MULTI_VIEW_CONTRACTION.value:
            reward, violations = self.verify_contraction_assessment(reasoning)
            metadata["sub_method"] = "contraction_assessment"
            metadata["violations"] = violations
            if not violations:
                if predicted_answer.strip().lower() != ground_truth.strip().lower():
                    reward = -1.0
                    violations.append(
                        f"Incorrect answer: '{predicted_answer}' vs '{ground_truth}'"
                    )
                    metadata["violations"] = violations
            return reward, metadata

        if category == VQACategory.LV_FUNCTION.value:
            reward, violations = self.verify_ef_consistency(reasoning)
            metadata["sub_method"] = "ef_consistency"
            metadata["violations"] = violations
            if not violations:
                if predicted_answer.strip().lower() != ground_truth.strip().lower():
                    reward = -1.0
                    violations.append(
                        f"Incorrect answer: '{predicted_answer}' vs '{ground_truth}'"
                    )
                    metadata["violations"] = violations
            return reward, metadata

        # Default: answer-match only
        metadata["sub_method"] = "answer_match"
        if predicted_answer.strip().lower() == ground_truth.strip().lower():
            return 1.0, metadata

        metadata["violations"] = [
            f"Incorrect answer: '{predicted_answer}' vs '{ground_truth}'"
        ]
        return -1.0, metadata


# =========================================================================
# Private helpers (module-level)
# =========================================================================


def _has_filling_pressure_evidence(
    metrics: dict[str, float | None], text: str
) -> bool:
    """Check for ACC/AHA 2022 filling pressure markers."""
    lavi = metrics.get("lavi")
    if lavi is not None and lavi >= 34:
        return True

    e_e_prime = metrics.get("e_e_prime")
    if e_e_prime is not None and e_e_prime >= 15:
        return True

    gls = metrics.get("gls")
    if gls is not None:
        # GLS is typically reported as a negative value; abnormal if |GLS| < 16
        if abs(gls) < 16:
            return True

    if re.search(r"abnormal\s+GLS", text, re.IGNORECASE):
        return True

    return False


def _has_structural_evidence(
    metrics: dict[str, float | None], text: str
) -> bool:
    """Check for ACC/AHA 2022 structural heart disease markers."""
    lvmi = metrics.get("lvmi")
    if lvmi is not None and lvmi >= 95:
        return True

    rwt = metrics.get("rwt")
    if rwt is not None and rwt > 0.42:
        return True

    lavi = metrics.get("lavi")
    if lavi is not None and lavi >= 34:
        return True

    structural_keywords = re.compile(
        r"(?:concentric\s+(?:remodel|hypertrophy)|eccentric\s+hypertrophy|"
        r"left\s+atrial\s+(?:enlargement|dilation)|"
        r"LV\s+hypertrophy|diastolic\s+dysfunction)",
        re.IGNORECASE,
    )
    if structural_keywords.search(text):
        return True

    return False
