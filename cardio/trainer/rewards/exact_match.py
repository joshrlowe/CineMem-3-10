"""Simple text-matching reward functions for GRPO training.

Ported from ``_reference/CineMem/main/trainer/rewards.py``.
"""

from __future__ import annotations

import re


def exact_match_reward(
    preds: list[str],
    refs: list[str | None],
) -> list[float]:
    """1.0 if whitespace-normalised prediction equals reference, else 0.0."""
    out: list[float] = []
    for p, r in zip(preds, refs):
        if r is None:
            out.append(0.0)
            continue
        p_norm = re.sub(r"\s+", " ", p.strip().lower())
        r_norm = re.sub(r"\s+", " ", r.strip().lower())
        out.append(1.0 if p_norm == r_norm else 0.0)
    return out


def substring_reward(
    preds: list[str],
    refs: list[str | None],
) -> list[float]:
    """1.0 if the reference appears as a substring of the prediction."""
    out: list[float] = []
    for p, r in zip(preds, refs):
        if r is None:
            out.append(0.0)
            continue
        out.append(1.0 if r.lower() in p.lower() else 0.0)
    return out
