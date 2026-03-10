"""Basic VLM evaluation (exact-match / substring).

Ported from ``_reference/CineMem/main/cli/eval.py``.  This module
exposes a single callable :func:`evaluate_vlm` that iterates a dataset,
generates predictions, and computes a basic text-matching score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from cardio.data.collate import collate_samples
from cardio.trainer.rewards.exact_match import exact_match_reward, substring_reward
from cardio.utils.logging import get_logger

if TYPE_CHECKING:
    from cardio.data.datasets.jsonl_vl import JsonlVLDataset
    from cardio.vlm.model import CineMemModel

logger = get_logger(__name__)


def evaluate_vlm(
    model: CineMemModel,
    dataset: JsonlVLDataset,
    metric: str = "substr",
    max_new_tokens: int = 256,
    enable_cinemem: bool = True,
) -> dict[str, Any]:
    """Run basic VLM evaluation on *dataset*.

    Args:
        model: a :class:`~cardio.vlm.model.CineMemModel` in eval mode.
        dataset: a :class:`~cardio.data.datasets.jsonl_vl.JsonlVLDataset`.
        metric: ``"exact"`` for exact-match, ``"substr"`` for substring.
        max_new_tokens: generation length budget.
        enable_cinemem: whether to activate memory-augmented decoding.

    Returns:
        Dict with ``score``, ``n_samples``, and ``per_sample`` results.
    """
    preds: list[str] = []
    refs: list[str] = []
    per_sample: list[dict[str, Any]] = []

    for i in tqdm(range(len(dataset)), desc="VLM eval"):
        sample = dataset[i]
        batch = collate_samples([sample])
        img = batch["images"][0]
        prompt = batch["prompts"][0]
        answer = batch["answers"][0]
        if answer is None:
            continue

        pred = model.generate(
            images=[img],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            enable_cinemem=enable_cinemem,
        )[0]
        preds.append(pred)
        refs.append(answer)

        per_sample.append({
            "id": sample.id,
            "prediction": pred[:200],
            "answer": answer[:200],
        })

    reward_fn = exact_match_reward if metric == "exact" else substring_reward
    rewards = reward_fn(preds, refs)
    score = sum(rewards) / max(1, len(rewards))

    for rec, r in zip(per_sample, rewards, strict=True):
        rec["reward"] = r

    logger.info("%s score: %.4f (%d samples)", metric, score, len(rewards))

    return {
        "score": score,
        "metric": metric,
        "n_samples": len(rewards),
        "per_sample": per_sample,
    }
