"""Cardiac-specific reward function for GRPO Stage 2.

Implements the 4-point scoring rubric:

    Answer correct        -> +1.0
    Memory/tool used      -> +0.5
    Complete process      -> +1.0
    List 2 candidates     -> +0.5
    No contradiction      -> +0.5
    Management suggestion -> +0.5
                           ------
    Maximum               -> 4.0

Normalised to ``[0, 1]`` for RL training via :func:`cardiac_reward_normalised`.

Ported from ``_reference/CineMem/main/trainer/cardiac_rewards.py`` with
dual-memory token support (TDM/PSM).
"""

from __future__ import annotations

import re

from cardio.vlm.constants import (
    LONG_END,
    LONG_INVOKE,
    PSM_END,
    PSM_INVOKE,
    SHORT_END,
    SHORT_INVOKE,
    TDM_END,
    TDM_INVOKE,
)

# ---------------------------------------------------------------------------
# Sub-component checks
# ---------------------------------------------------------------------------


def _check_correct_option(pred: str, ref: str) -> float:
    p = re.sub(r"\s+", " ", pred.strip().lower())
    r = re.sub(r"\s+", " ", ref.strip().lower())

    if p == r:
        return 1.0

    key_numbers = re.findall(r"[\d]+\.?\d*", r)
    if key_numbers:
        matches = sum(1 for n in key_numbers if n in p)
        if matches >= len(key_numbers) * 0.6:
            return 0.8

    keywords = [w for w in r.split() if len(w) > 3]
    if keywords:
        overlap = sum(1 for w in keywords if w in p) / len(keywords)
        if overlap > 0.5:
            return 0.5

    return 0.0


def _check_memory_used(raw: str) -> float:
    has_short = SHORT_INVOKE in raw and SHORT_END in raw
    has_long = LONG_INVOKE in raw and LONG_END in raw
    has_tdm = TDM_INVOKE in raw and TDM_END in raw
    has_psm = PSM_INVOKE in raw and PSM_END in raw
    has_tool = "<tool_call>" in raw

    has_memory = has_short or has_long or has_tdm or has_psm
    if has_memory and has_tool:
        return 0.5
    if has_memory or has_tool:
        return 0.3
    return 0.0


def _check_complete_reasoning(pred: str) -> float:
    sections = {
        "observe": bool(re.search(r"<step>OBSERVE</step>", pred, re.IGNORECASE)),
        "measure": bool(re.search(r"<step>MEASURE</step>", pred, re.IGNORECASE)),
        "reason": bool(re.search(r"<step>REASON</step>", pred, re.IGNORECASE)),
        "conclude": bool(re.search(r"<step>CONCLUDE</step>", pred, re.IGNORECASE)),
    }
    present = sum(sections.values())
    if present >= 4:
        return 1.0
    if present >= 3:
        return 0.7
    if present >= 2:
        return 0.4
    return 0.0


def _check_differential_candidates(pred: str) -> float:
    patterns = [
        r"differential",
        r"candidates?\s*:",
        r"(?:could|may)\s+(?:also\s+)?(?:be|indicate|suggest)",
        r"(?:versus|vs\.?)\s+\w+",
        r"(?:alternatively|another\s+possibility)",
    ]
    matches = sum(1 for p in patterns if re.search(p, pred, re.IGNORECASE))
    if matches >= 2:
        return 0.5
    if matches >= 1:
        return 0.25
    return 0.0


def _check_no_contradiction(pred: str) -> float:
    severity_map = {
        "severely reduced": (0, 35),
        "moderately reduced": (35, 45),
        "mildly reduced": (45, 55),
        "normal": (55, 100),
    }
    ef_vals = re.findall(
        r"(?:EF|ejection fraction)[^\d]*?([\d]+\.?\d*)\s*%", pred, re.IGNORECASE
    )
    severities = [s for s in severity_map if s in pred.lower()]

    if ef_vals and severities:
        try:
            ef = float(ef_vals[0])
            for sev in severities:
                lo, hi = severity_map[sev]
                if lo <= ef < hi:
                    return 0.5
            return 0.0
        except ValueError:
            pass

    return 0.5


def _check_management_suggestion(pred: str) -> float:
    patterns = [
        r"(?:consider|recommend|suggest|advise)",
        r"(?:follow[- ]up|surveillance|monitoring)",
        r"(?:escalat|intensif|optimiz)",
        r"(?:clinical\s+(?:action|note|implication))",
        r"(?:therapy|treatment|intervention|management)",
    ]
    if any(re.search(p, pred, re.IGNORECASE) for p in patterns):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_SCORE = 4.0


def cardiac_reward(
    preds: list[str],
    refs: list[str | None],
    raw_outputs: list[str] | None = None,
) -> list[float]:
    """Compute per-sample cardiac reward (raw score 0-4)."""
    if raw_outputs is None:
        raw_outputs = preds

    rewards: list[float] = []
    for pred, ref, raw in zip(preds, refs, raw_outputs):
        if ref is None:
            rewards.append(0.0)
            continue
        score = (
            _check_correct_option(pred, ref)
            + _check_memory_used(raw)
            + _check_complete_reasoning(pred)
            + _check_differential_candidates(pred)
            + _check_no_contradiction(pred)
            + _check_management_suggestion(pred)
        )
        rewards.append(score)
    return rewards


def cardiac_reward_normalised(
    preds: list[str],
    refs: list[str | None],
    raw_outputs: list[str] | None = None,
) -> list[float]:
    """Compute per-sample cardiac reward normalised to ``[0, 1]``."""
    return [r / MAX_SCORE for r in cardiac_reward(preds, refs, raw_outputs)]
