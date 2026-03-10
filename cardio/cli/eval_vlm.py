r"""CLI entry point for VLM evaluation (``cardio-eval-vlm``).

Supports two evaluation modes:

* ``basic`` -- exact-match or substring scoring via
  :func:`~cardio.eval.vlm.evaluate_vlm`.
* ``cardiac`` -- cardiac rubric + optional CompositeRewardEngine
  breakdown via :func:`~cardio.eval.vlm_cardiac.evaluate_vlm_cardiac`.

Usage::

    cardio-eval-vlm \
        --jsonl data/vqa_jsonl/test.jsonl \
        --image-dir /path/to/images \
        --mode cardiac \
        --enable_cinemem \
        --composite \
        --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from cardio.data.datasets.jsonl_vl import JsonlVLDataset
from cardio.utils import get_logger, to_torch_dtype
from cardio.vlm import CineMemModel, load_qwen25vl
from cardio.vlm.config import build_cinemem_config

if TYPE_CHECKING:
    from cardio.trainer.rewards.composite import CompositeRewardEngine

logger = get_logger(__name__)

_w = sys.stdout.write


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    ap = argparse.ArgumentParser(
        description="Evaluate a CineMemModel on a VQA dataset.",
    )
    ap.add_argument("--config", default="configs/cinemem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--ckpt", default=None, help="Folder containing main.pt")
    ap.add_argument("--jsonl", required=True, help="Evaluation dataset JSONL")
    ap.add_argument("--image-dir", default="")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--enable_cinemem", action="store_true")
    ap.add_argument(
        "--mode",
        choices=["basic", "cardiac"],
        default="cardiac",
    )
    ap.add_argument("--metric", choices=["exact", "substr"], default="substr")
    ap.add_argument("--output", default=None, help="Write per-sample JSONL")
    ap.add_argument(
        "--composite",
        action="store_true",
        help="Run CompositeRewardEngine (cardiac mode only)",
    )
    return ap.parse_args()


def _load_model(
    args: argparse.Namespace,
) -> CineMemModel:
    """Load the CineMemModel from config + optional checkpoint."""
    from cardio.utils.misc import load_yaml

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_cinemem_config(cfg_dict)

    model_name: str = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map: str = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(
        model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust,
    )
    cinemem = CineMemModel(base_model, tokenizer, processor, viscfg)

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt) / "main.pt"
        state = torch.load(ckpt_path, map_location="cpu")
        cinemem.load_state_dict(state["cinemem_state"], strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)

    cinemem.eval()
    return cinemem


def _build_composite_engine() -> CompositeRewardEngine:
    """Construct a default CompositeRewardEngine."""
    from cardio.trainer.rewards.cardiac import cardiac_reward_normalised
    from cardio.trainer.rewards.composite import CompositeRewardEngine as _Engine
    from cardio.trainer.rewards.dcr import AutoMetricConverter, DivideConquerEvaluator
    from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier
    from cardio.trainer.rewards.vprm import ACCAHAVerifier

    return _Engine(
        cardiac_verifier=cardiac_reward_normalised,
        acc_aha_verifier=ACCAHAVerifier(),
        memory_verifier=MemoryInvocationVerifier(),
        dcr_evaluator=DivideConquerEvaluator(),
        amc=AutoMetricConverter(),
    )


def _print_basic(result: dict) -> None:
    """Pretty-print basic evaluation results."""
    lines = [
        "\n" + "=" * 60,
        "VLM EVALUATION RESULTS",
        "=" * 60,
        f"  Metric:    {result['metric']}",
        f"  Score:     {result['score']:.4f}",
        f"  Samples:   {result['n_samples']}",
        "=" * 60,
    ]
    _w("\n".join(lines) + "\n")


def _print_cardiac(result: dict) -> None:
    """Pretty-print cardiac evaluation results."""
    overall = result["overall"]
    lines = [
        "\n" + "=" * 60,
        "CARDIAC EVALUATION RESULTS",
        "=" * 60,
        f"\nOverall ({overall['n_samples']} samples):",
        f"  Cardiac reward (norm): {overall['cardiac_norm']:.4f}",
        f"  Exact match:           {overall['exact_match']:.4f}",
        "\nPer-category breakdown:",
    ]
    for cat, vals in sorted(result["per_category"].items()):
        lines.append(
            f"  {cat:40s}  n={vals['n']:4d}  "
            f"cardiac={vals['cardiac_norm']:.4f}  "
            f"exact={vals['exact_match']:.4f}"
        )

    if result.get("multiview_categories"):
        lines.append("\nMulti-view categories (7-12):")
        for cat, vals in sorted(result["multiview_categories"].items()):
            lines.append(
                f"  {cat:40s}  n={vals['n']:4d}  "
                f"cardiac={vals['cardiac_norm']:.4f}  "
                f"exact={vals['exact_match']:.4f}"
            )

    if "composite_breakdown" in result:
        cb = result["composite_breakdown"]
        lines.extend([
            "\nComposite reward breakdown:",
            f"  Composite mean: {cb['composite_mean']:.4f}",
            f"  VPRM mean:      {cb['vprm_mean']:.4f}",
            f"  Memory mean:    {cb['memory_mean']:.4f}",
            f"  DCR mean:       {cb['dcr_mean']:.4f}",
        ])

    lines.append("=" * 60)
    _w("\n".join(lines) + "\n")


def _write_jsonl(per_sample: list[dict], output_path: str) -> None:
    """Write per-sample results to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in per_sample:
            serialisable = {
                k: v for k, v in rec.items()
                if not isinstance(v, (torch.Tensor,))
            }
            f.write(json.dumps(serialisable, ensure_ascii=False) + "\n")
    logger.info("Results written to %s", output_path)


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    model = _load_model(args)
    dataset = JsonlVLDataset(args.jsonl, image_dir=args.image_dir)

    if args.mode == "basic":
        from cardio.eval.vlm import evaluate_vlm

        result = evaluate_vlm(
            model=model,
            dataset=dataset,
            metric=args.metric,
            max_new_tokens=args.max_new_tokens,
            enable_cinemem=args.enable_cinemem,
        )
        _print_basic(result)
        per_sample = result["per_sample"]

    else:
        from cardio.eval.vlm_cardiac import evaluate_vlm_cardiac

        engine = _build_composite_engine() if args.composite else None
        result = evaluate_vlm_cardiac(
            model=model,
            dataset=dataset,
            composite_engine=engine,
            max_new_tokens=args.max_new_tokens,
            enable_cinemem=args.enable_cinemem,
        )
        _print_cardiac(result)
        per_sample = result["per_sample"]

    if args.output:
        _write_jsonl(per_sample, args.output)


if __name__ == "__main__":
    main()
