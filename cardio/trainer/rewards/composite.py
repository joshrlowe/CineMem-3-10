"""Composite reward engine for TR-GRPO training.

Aggregates four reward signals into a single composite reward and an
optional per-token penalty mask:

1. **Cardiac rubric** -- 4-point answer-quality scale normalised to [0, 1].
2. **ACC/AHA VPRM** -- clinical logic verification against ACC/AHA 2022.
3. **Memory invocation penalty** -- verifies TDM/PSM usage backs claims.
4. **DCR visual grounding** -- bounding-box IoU for anatomical claims.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cardio.trainer.rewards.dcr import AutoMetricConverter, DivideConquerEvaluator
from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier
from cardio.trainer.rewards.vprm import ACCAHAVerifier

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "cardiac": 0.3,
    "vprm": 0.3,
    "memory": 0.2,
    "dcr": 0.2,
}


# =========================================================================
# CompositeRewardEngine
# =========================================================================


class CompositeRewardEngine:
    """Aggregate all reward signals into a composite reward for NGRPO.

    Args:
        cardiac_verifier: callable matching the signature of
            :func:`~cardio.trainer.rewards.cardiac.cardiac_reward_normalised`
            (``(preds, refs, raw_outputs) -> list[float]``).
        acc_aha_verifier: an :class:`ACCAHAVerifier` instance.
        memory_verifier: a :class:`MemoryInvocationVerifier` instance.
        dcr_evaluator: a :class:`DivideConquerEvaluator` instance.
        amc: an :class:`AutoMetricConverter` instance.
        weights: per-component weight overrides.  Missing keys fall back
            to :data:`DEFAULT_WEIGHTS`.
    """

    def __init__(
        self,
        cardiac_verifier: Callable[
            [list[str], list[str | None], list[str] | None], list[float]
        ],
        acc_aha_verifier: ACCAHAVerifier,
        memory_verifier: MemoryInvocationVerifier,
        dcr_evaluator: DivideConquerEvaluator,
        amc: AutoMetricConverter,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.cardiac_verifier = cardiac_verifier
        self.acc_aha_verifier = acc_aha_verifier
        self.memory_verifier = memory_verifier
        self.dcr_evaluator = dcr_evaluator
        self.amc = amc

        w = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            w.update(weights)
        self.weights = w

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        generated_text: str,
        predicted_answer: str,
        ground_truth: str,
        category: str,
        invocation_log: list[dict],
        predicted_bboxes: dict[str, list[float]] | None,
        ground_truth_masks: dict | None,
        token_offsets: list[tuple[int, int]] | None,
        seq_len: int | None = None,
    ) -> dict:
        """Compute the composite reward and per-token penalty mask.

        Returns a dict with keys ``composite_reward``,
        ``token_penalty_mask``, ``component_rewards``, ``violations``,
        and ``metadata``.
        """
        all_violations: list[str] = []
        metadata: dict = {}

        # --- 1. Cardiac rubric (score in [0, 1]) -------------------------
        cardiac_score = self._compute_cardiac(
            generated_text, ground_truth, all_violations, metadata
        )

        # --- 2. ACC/AHA VPRM (reward in [-1, 1] -> remap to [0, 1]) ------
        vprm_score = self._compute_vprm(
            generated_text, predicted_answer, ground_truth, category,
            all_violations, metadata,
        )

        # --- 3. Memory penalty (normalise to [0, 1]) ---------------------
        memory_score = self._compute_memory(
            generated_text, invocation_log, token_offsets,
            all_violations, metadata,
        )

        # --- 4. DCR grounding (score in [0, 1], may produce token mask) --
        dcr_score, token_mask = self._compute_dcr(
            generated_text, predicted_bboxes, ground_truth_masks,
            token_offsets, seq_len, all_violations, metadata,
        )

        # --- Weighted composite ------------------------------------------
        composite = (
            self.weights["cardiac"] * cardiac_score
            + self.weights["vprm"] * vprm_score
            + self.weights["memory"] * memory_score
            + self.weights["dcr"] * dcr_score
        )
        composite = max(0.0, min(1.0, composite))

        return {
            "composite_reward": composite,
            "token_penalty_mask": token_mask,
            "component_rewards": {
                "cardiac": cardiac_score,
                "vprm": vprm_score,
                "memory": memory_score,
                "dcr": dcr_score,
            },
            "violations": all_violations,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def _compute_cardiac(
        self,
        generated_text: str,
        ground_truth: str,
        violations: list[str],
        metadata: dict,
    ) -> float:
        scores = self.cardiac_verifier(
            [generated_text], [ground_truth], [generated_text]
        )
        score = scores[0]
        metadata["cardiac_raw"] = score
        return score

    def _compute_vprm(
        self,
        generated_text: str,
        predicted_answer: str,
        ground_truth: str,
        category: str,
        violations: list[str],
        metadata: dict,
    ) -> float:
        reward, vprm_meta = self.acc_aha_verifier.compute_reward(
            generated_text, predicted_answer, ground_truth, category
        )
        metadata["vprm"] = vprm_meta

        vprm_violations = vprm_meta.get("violations", [])
        if isinstance(vprm_violations, list):
            violations.extend(str(v) for v in vprm_violations)

        return (reward + 1.0) / 2.0

    def _compute_memory(
        self,
        generated_text: str,
        invocation_log: list[dict],
        token_offsets: list[tuple[int, int]] | None,
        violations: list[str],
        metadata: dict,
    ) -> float:
        if token_offsets is None:
            metadata["memory_skipped"] = True
            return 1.0

        start_offsets = [s for s, _ in token_offsets]
        claims = self.memory_verifier.find_clinical_claims(
            generated_text, start_offsets
        )
        penalty, mem_violations = self.memory_verifier.verify_invocations(
            claims, invocation_log
        )

        for v in mem_violations:
            violations.append(
                f"Memory: missing {v.get('expected_memory', '?')} "
                f"for \"{v.get('term', '?')}\""
            )

        metadata["memory_claims"] = len(claims)
        metadata["memory_violations"] = len(mem_violations)

        n_claims = max(len(claims), 1)
        return max(0.0, 1.0 + penalty / n_claims)

    def _compute_dcr(
        self,
        generated_text: str,
        predicted_bboxes: dict[str, list[float]] | None,
        ground_truth_masks: dict | None,
        token_offsets: list[tuple[int, int]] | None,
        seq_len: int | None,
        violations: list[str],
        metadata: dict,
    ) -> tuple[float, torch.Tensor | None]:
        if predicted_bboxes is None or ground_truth_masks is None:
            metadata["dcr_skipped"] = True
            return 1.0, None

        claims = self.dcr_evaluator.isolate_claims(generated_text)
        if not claims:
            metadata["dcr_claims"] = 0
            return 1.0, None

        evaluated = self.dcr_evaluator.evaluate_claims(
            claims, predicted_bboxes, ground_truth_masks
        )

        for claim in evaluated:
            penalty = self.amc.compute_penalty(claim["iou"])
            claim["penalty"] = penalty

        penalties = [c["penalty"] for c in evaluated]
        dcr_score = 1.0 - (sum(penalties) / len(penalties))

        hallucinated = [
            c for c in evaluated if c.get("grounding_status") == "hallucination"
        ]
        for h in hallucinated:
            violations.append(
                f"DCR: hallucinated grounding for \"{h.get('anatomy', '?')}\" "
                f"(IoU={h.get('iou', 0):.2f})"
            )

        metadata["dcr_claims"] = len(evaluated)
        metadata["dcr_mean_iou"] = (
            sum(c["iou"] for c in evaluated) / len(evaluated)
        )

        token_mask: torch.Tensor | None = None
        if token_offsets is not None and seq_len is not None:
            token_mask = self.amc.build_token_mask(
                evaluated, token_offsets, seq_len
            )

        return dcr_score, token_mask
