"""Memory invocation penalty for cardiac GRPO training.

Verifies that the model properly invoked the dual-memory system (TDM / PSM)
when its generated reasoning contains clinical claims that depend on those
memory pathways.

* **Motion-related claims** (contraction, wall motion, EF, …) require a
  preceding TDM invocation (``<tdm_I>`` or legacy ``<ms_I>``).
* **Structure-related claims** (dilation, hypertrophy, chamber size, …)
  require a preceding PSM invocation (``<psm_I>`` or legacy ``<ml_I>``).

The penalty is computed per-claim: if no matching invocation is found within
a configurable token-distance window before the claim, a ``-1.0`` penalty is
assessed.  Claims that *are* backed by a nearby invocation receive no penalty.
"""

from __future__ import annotations

import bisect
import re

# ---------------------------------------------------------------------------
# Clinical term sets
# ---------------------------------------------------------------------------

MOTION_TERMS: set[str] = {
    "hypokinesis",
    "akinesis",
    "dyskinesis",
    "contraction",
    "systolic",
    "diastolic",
    "wall motion",
    "ejection fraction",
    "EF",
    "thickening",
    "thinning",
    "TAPSE",
    "strain",
    "shortening",
}

STRUCTURE_TERMS: set[str] = {
    "dilation",
    "dilated",
    "hypertrophy",
    "hypertrophic",
    "thickness",
    "diameter",
    "volume",
    "geometry",
    "remodeling",
    "chamber size",
    "septal",
    "aneurysm",
}

# TDM / legacy-short types logged by CineMemModel.generate()
_TDM_TYPES: frozenset[str] = frozenset({"tdm", "short"})
# PSM / legacy-long types
_PSM_TYPES: frozenset[str] = frozenset({"psm", "long"})

# Pre-compiled pattern that matches any clinical term (longest-first so
# multi-word terms like "wall motion" match before single-word fragments).
_ALL_TERMS: dict[str, str] = {}
for _t in MOTION_TERMS:
    _ALL_TERMS[_t] = "motion"
for _t in STRUCTURE_TERMS:
    _ALL_TERMS[_t] = "structure"

_TERM_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(_ALL_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Sentence boundary (used for the "sentence" field in claim dicts).
_SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]?")


def _char_offset_to_token_idx(char_pos: int, token_offsets: list[int]) -> int:
    """Map a character position to the enclosing token index.

    ``token_offsets[i]`` is the character offset where the *i*-th token
    begins.  We find the rightmost token whose start offset is ``<=``
    *char_pos* (i.e. the token that contains or immediately precedes the
    character).
    """
    idx = bisect.bisect_right(token_offsets, char_pos) - 1
    return max(idx, 0)


def _sentence_around(text: str, pos: int) -> str:
    """Return the sentence containing character position *pos*."""
    for m in _SENT_RE.finditer(text):
        if m.start() <= pos < m.end():
            return m.group(0).strip()
    return text[max(0, pos - 40) : pos + 40].strip()


# =========================================================================
# Verifier
# =========================================================================


class MemoryInvocationVerifier:
    """Check that memory invocations back the clinical claims in generated text.

    For each motion-related claim, a TDM (or legacy short-memory) invocation
    must appear within *context_window* tokens **before** the claim.  For each
    structure-related claim, a PSM (or legacy long-memory) invocation must
    appear similarly.

    Args:
        context_window: maximum token distance before a claim where a
            matching memory invocation is still considered valid.
    """

    def __init__(self, context_window: int = 50) -> None:
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Claim scanning
    # ------------------------------------------------------------------

    def find_clinical_claims(
        self,
        text: str,
        token_offsets: list[int],
    ) -> list[dict]:
        """Scan *text* for motion and structure terms.

        Args:
            text: the generated reasoning text (decoded tokens).
            token_offsets: list whose *i*-th element is the character
                offset where the *i*-th token starts.

        Returns:
            A list of claim dicts, each with keys ``"term"``,
            ``"type"`` (``"motion"`` or ``"structure"``),
            ``"token_idx"`` (index into the token sequence), and
            ``"sentence"`` (surrounding sentence fragment).
        """
        claims: list[dict] = []
        seen_spans: set[tuple[int, int]] = set()

        for m in _TERM_RE.finditer(text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)

            matched_term = m.group(0)
            term_key = matched_term.lower()

            claim_type: str | None = None
            for canonical, ctype in _ALL_TERMS.items():
                if canonical.lower() == term_key:
                    claim_type = ctype
                    break
            if claim_type is None:
                continue

            claims.append({
                "term": matched_term,
                "type": claim_type,
                "token_idx": _char_offset_to_token_idx(m.start(), token_offsets),
                "sentence": _sentence_around(text, m.start()),
            })

        return claims

    # ------------------------------------------------------------------
    # Invocation verification
    # ------------------------------------------------------------------

    def verify_invocations(
        self,
        claims: list[dict],
        invocation_log: list[dict],
    ) -> tuple[float, list[dict]]:
        """Cross-reference claims against the invocation log.

        For each claim, the nearest **preceding** invocation of the
        correct type (TDM for motion, PSM for structure) is located.
        If it falls within :attr:`context_window` tokens, no penalty is
        applied; otherwise a ``-1.0`` penalty is assessed for that claim.

        Args:
            claims: output of :meth:`find_clinical_claims`.
            invocation_log: the ``invocation_log`` list produced by
                :meth:`~cardio.vlm.model.CineMemModel.generate`.  Each
                entry has at least ``{"step": int, "type": str}``.

        Returns:
            ``(penalty_sum, violation_details)`` where *penalty_sum* is
            ``<= 0`` (sum of ``-1.0`` per violation) and
            *violation_details* is a list of dicts describing each
            unmatched claim.
        """
        if not claims:
            return 0.0, []

        tdm_steps = sorted(
            entry["step"] for entry in invocation_log if entry.get("type") in _TDM_TYPES
        )
        psm_steps = sorted(
            entry["step"] for entry in invocation_log if entry.get("type") in _PSM_TYPES
        )

        penalty = 0.0
        violations: list[dict] = []

        for claim in claims:
            claim_tok = claim["token_idx"]
            required = claim["type"]

            steps = tdm_steps if required == "motion" else psm_steps
            expected_mem = "TDM" if required == "motion" else "PSM"

            nearest = self._find_nearest_preceding(claim_tok, steps)

            if nearest is None or (claim_tok - nearest) > self.context_window:
                penalty -= 1.0
                detail: dict = {
                    "term": claim["term"],
                    "type": required,
                    "token_idx": claim_tok,
                    "sentence": claim["sentence"],
                    "expected_memory": expected_mem,
                    "nearest_invocation_step": nearest,
                }
                if nearest is not None:
                    detail["distance"] = claim_tok - nearest
                violations.append(detail)

        return penalty, violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_nearest_preceding(claim_token: int, sorted_steps: list[int]) -> int | None:
        """Return the largest step in *sorted_steps* that is ``<= claim_token``."""
        idx = bisect.bisect_right(sorted_steps, claim_token) - 1
        if idx < 0:
            return None
        return sorted_steps[idx]
