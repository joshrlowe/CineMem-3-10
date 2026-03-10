"""Cardiac-specific VLM evaluation with CompositeRewardEngine integration.

Ported from ``_reference/CineMem/main/cli/eval_cardiac.py`` with the
following extensions:

* **CompositeRewardEngine** breakdown -- per-sample VPRM, memory
  compliance, and DCR grounding scores.
* **Per-category reporting for Categories 7-12** -- the multi-view VQA
  categories defined in :mod:`cardio.data.vqa`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from cardio.data.collate import collate_samples
from cardio.eval.metric import aggregate_scores
from cardio.trainer.rewards.cardiac import MAX_SCORE, cardiac_reward
from cardio.trainer.rewards.exact_match import exact_match_reward, substring_reward
from cardio.utils.logging import get_logger

if TYPE_CHECKING:
    from cardio.data.datasets.jsonl_vl import JsonlVLDataset
    from cardio.trainer.rewards.composite import CompositeRewardEngine
    from cardio.vlm.model import CineMemModel

logger = get_logger(__name__)

MULTI_VIEW_CATEGORIES = frozenset({
    "cross_view_structure",
    "multi_view_contraction",
    "rv_assessment",
    "multi_phase_functional_summary",
    "cross_view_consistency_verification",
    "temporal_contraction_pattern",
})


def evaluate_vlm_cardiac(
    model: CineMemModel,
    dataset: JsonlVLDataset,
    composite_engine: CompositeRewardEngine | None = None,
    max_new_tokens: int = 256,
    enable_cinemem: bool = True,
) -> dict[str, Any]:
    """Run cardiac evaluation with optional composite-reward breakdown.

    Args:
        model: a :class:`~cardio.vlm.model.CineMemModel` in eval mode.
        dataset: evaluation dataset.
        composite_engine: if provided, each sample is also scored by
            :meth:`CompositeRewardEngine.compute`.
        max_new_tokens: generation length budget.
        enable_cinemem: whether to activate memory-augmented decoding.

    Returns:
        Dict with keys ``overall``, ``per_category``,
        ``composite_breakdown`` (if engine supplied), and ``per_sample``.
    """
    results: list[dict[str, Any]] = []
    category_cardiac: dict[str, list[float]] = defaultdict(list)
    category_exact: dict[str, list[float]] = defaultdict(list)

    composite_vprm: list[float] = []
    composite_memory: list[float] = []
    composite_dcr: list[float] = []
    composite_total: list[float] = []

    for i in tqdm(range(len(dataset)), desc="Cardiac eval"):
        sample = dataset[i]
        batch = collate_samples([sample])
        img = batch["images"][0]
        prompt = batch["prompts"][0]
        answer = batch["answers"][0]
        if answer is None:
            continue

        pred_raw = model.generate(
            images=[img],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            enable_cinemem=enable_cinemem,
            skip_special_tokens=False,
        )[0]

        pred_clean = model.generate(
            images=[img],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            enable_cinemem=enable_cinemem,
            skip_special_tokens=True,
        )[0]

        cr = cardiac_reward([pred_clean], [answer], [pred_raw])[0]
        cr_norm = cr / MAX_SCORE
        em = exact_match_reward([pred_clean], [answer])[0]
        sub = substring_reward([pred_clean], [answer])[0]

        category = sample.meta.get("category", "unknown") if sample.meta else "unknown"
        category_cardiac[category].append(cr_norm)
        category_exact[category].append(em)

        record: dict[str, Any] = {
            "id": sample.id,
            "category": category,
            "cardiac_reward": cr,
            "cardiac_reward_norm": cr_norm,
            "exact_match": em,
            "substring_match": sub,
            "prediction": pred_clean[:200],
        }

        if composite_engine is not None:
            comp_result = composite_engine.compute(
                generated_text=pred_raw,
                predicted_answer=pred_clean,
                ground_truth=answer,
                category=category,
                invocation_log=[],
                predicted_bboxes=None,
                ground_truth_masks=None,
                token_offsets=None,
                seq_len=None,
            )
            comp_rewards = comp_result["component_rewards"]
            record["composite_reward"] = comp_result["composite_reward"]
            record["vprm"] = comp_rewards["vprm"]
            record["memory_compliance"] = comp_rewards["memory"]
            record["dcr_grounding"] = comp_rewards["dcr"]
            record["violations"] = comp_result["violations"]

            composite_vprm.append(comp_rewards["vprm"])
            composite_memory.append(comp_rewards["memory"])
            composite_dcr.append(comp_rewards["dcr"])
            composite_total.append(comp_result["composite_reward"])

        results.append(record)

    # ----- Overall summary ------------------------------------------------
    all_cr = [r["cardiac_reward_norm"] for r in results]
    all_em = [r["exact_match"] for r in results]
    overall: dict[str, Any] = {
        "cardiac_norm": sum(all_cr) / max(1, len(all_cr)),
        "exact_match": sum(all_em) / max(1, len(all_em)),
        "n_samples": len(results),
    }

    # ----- Per-category summary -------------------------------------------
    cardiac_agg = aggregate_scores(category_cardiac)
    exact_agg = aggregate_scores(category_exact)

    per_category: dict[str, dict[str, Any]] = {}
    for cat in sorted({*category_cardiac, *category_exact}):
        per_category[cat] = {
            "n": cardiac_agg["per_category"].get(cat, {}).get("n", 0),
            "cardiac_norm": cardiac_agg["per_category"].get(cat, {}).get("mean", 0.0),
            "exact_match": exact_agg["per_category"].get(cat, {}).get("mean", 0.0),
        }

    # ----- Multi-view categories 7-12 highlight ---------------------------
    multiview_report: dict[str, dict[str, Any]] = {
        cat: per_category[cat]
        for cat in sorted(per_category)
        if cat in MULTI_VIEW_CATEGORIES
    }

    # ----- Composite breakdown (if engine was used) -----------------------
    composite_breakdown: dict[str, float] | None = None
    if composite_engine is not None and composite_total:
        n = len(composite_total)
        composite_breakdown = {
            "composite_mean": sum(composite_total) / n,
            "vprm_mean": sum(composite_vprm) / n,
            "memory_mean": sum(composite_memory) / n,
            "dcr_mean": sum(composite_dcr) / n,
        }

    output: dict[str, Any] = {
        "overall": overall,
        "per_category": per_category,
        "multiview_categories": multiview_report,
        "per_sample": results,
    }
    if composite_breakdown is not None:
        output["composite_breakdown"] = composite_breakdown

    return output
