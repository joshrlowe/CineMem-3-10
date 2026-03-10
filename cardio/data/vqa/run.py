"""Orchestrator for the VQA generation pipeline.

Usage:
    # Run all stages from HuggingFace (default)
    cardio-generate-vqa --output-dir ./vqa_output --splits train val test

    # Run all stages from local preprocessed data
    cardio-generate-vqa --source local --data-dir ~/.cache/cinema_datasets --output-dir ./vqa_output

    # Run individual stage
    cardio-generate-vqa --stage ground_truth --output-dir ./vqa_output

    # Emit all three formats during conversion
    cardio-generate-vqa --stage convert --format all --output-dir ./vqa_output

    # Push the final dataset to HuggingFace
    cardio-generate-vqa --stage convert --push-to-hub --hf-username myuser
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from cardio.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = ["ground_truth", "reasoning", "validate", "paraphrase", "convert"]
FORMATS = ["sharegpt", "jsonl", "vismem", "all"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="VQA generation pipeline for cardiac cine MRI datasets.",
    )

    # --- Data source ---
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "huggingface"],
        default="huggingface",
        help="Data source: 'huggingface' downloads from HF Hub, 'local' uses --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "cinema_datasets",
        help="Root directory containing {dataset}/processed/ subdirectories (local mode only).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory. Defaults to ~/.cache/huggingface/hub.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["acdc", "mnms", "mnms2"],
        help="Datasets to process (default: acdc mnms mnms2).",
    )

    # --- Split control ---
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test).",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vqa_output"),
        help="Output directory for generated VQA data.",
    )

    # --- Pipeline control ---
    parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default=None,
        help="Run a specific stage only. If omitted, runs all stages sequentially.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. Defaults to $OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for reasoning/paraphrasing (default: gpt-4o).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # --- Format conversion ---
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        choices=FORMATS,
        default=["sharegpt"],
        help="Output format(s) for the convert stage (default: sharegpt).",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for resolving image paths (used by vismem format).",
    )

    # --- HuggingFace upload ---
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the final VQA dataset to HuggingFace after conversion.",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default=None,
        help="HuggingFace username for the upload repo (e.g., 'myuser/cardiac_cine_vqa').",
    )

    return parser.parse_args(argv)


def _resolve_formats(raw: list[str]) -> set[str]:
    """Expand ``"all"`` into the concrete format names."""
    if "all" in raw:
        return {"sharegpt", "jsonl", "vismem"}
    return set(raw)


def _resolve_dataset_dirs(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve dataset name -> local root path based on the source."""
    if args.source == "huggingface":
        from cardio.data.vqa.hf_loader import download_datasets

        logger.info("Downloading datasets from HuggingFace Hub ...")
        return download_datasets(
            datasets=args.datasets,
            cache_dir=args.hf_cache_dir,
        )

    from cardio.data.vqa import DATASET_ACDC, DATASET_MNMS, DATASET_MNMS2

    all_local = {
        DATASET_ACDC: args.data_dir / "acdc" / "processed",
        DATASET_MNMS: args.data_dir / "mnms" / "processed",
        DATASET_MNMS2: args.data_dir / "mnms2" / "processed",
    }
    return {k: v for k, v in all_local.items() if k in args.datasets}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_ground_truth(args: argparse.Namespace, dataset_dirs: dict[str, Path]) -> dict[str, Path]:
    """Run Stage 1: Ground truth generation for all requested splits."""
    from cardio.data.vqa.ground_truth import generate_ground_truth

    logger.info("=== Stage 1: Ground Truth Generation ===")
    return generate_ground_truth(
        output_dir=args.output_dir,
        dataset_dirs=dataset_dirs,
        splits=args.splits,
        seed=args.seed,
    )


def run_reasoning(args: argparse.Namespace) -> Path:
    """Run Stage 2: GPT-4o reasoning generation (train split only)."""
    from cardio.data.vqa.reasoning import generate_reasoning

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OpenAI API key required for reasoning stage. Set --api-key or $OPENAI_API_KEY."
        raise ValueError(msg)

    gt_path = args.output_dir / "ground_truth_train.jsonl"
    if not gt_path.exists():
        msg = f"Ground truth file not found: {gt_path}. Run ground_truth stage first."
        raise FileNotFoundError(msg)

    logger.info("=== Stage 2: GPT-4o Reasoning Generation (train) ===")
    return generate_reasoning(
        ground_truth_path=gt_path,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
    )


def run_validate(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run Stage 3: Validation (train split only)."""
    from cardio.data.vqa.validate import validate_reasoning

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    reasoning_path = args.output_dir / "reasoning_raw.jsonl"
    if not reasoning_path.exists():
        msg = f"Reasoning file not found: {reasoning_path}. Run reasoning stage first."
        raise FileNotFoundError(msg)

    logger.info("=== Stage 3: Validation (train) ===")
    return validate_reasoning(
        reasoning_path=reasoning_path,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
    )


def run_paraphrase(args: argparse.Namespace) -> Path:
    """Run Stage 4: Question paraphrasing (train split only)."""
    from cardio.data.vqa.paraphrase import generate_paraphrases

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OpenAI API key required for paraphrase stage. Set --api-key or $OPENAI_API_KEY."
        raise ValueError(msg)

    canonical_path = args.output_dir / "canonical_vqa.jsonl"
    if not canonical_path.exists():
        msg = f"Canonical VQA file not found: {canonical_path}. Run validate stage first."
        raise FileNotFoundError(msg)

    logger.info("=== Stage 4: Question Paraphrasing (train) ===")
    return generate_paraphrases(
        canonical_path=canonical_path,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
    )


def run_convert(args: argparse.Namespace) -> list[Path]:
    """Run Stage 5: Format conversion to one or more output formats."""
    from cardio.data.vqa.convert import (
        convert_eval_splits,
        convert_to_jsonl_file,
        convert_to_llamafactory,
        convert_to_vismem_file,
    )

    formats = _resolve_formats(args.format)
    paths: list[Path] = []

    paraphrased_path = args.output_dir / "canonical_vqa_paraphrased.jsonl"
    has_paraphrased = paraphrased_path.exists()

    if not has_paraphrased:
        logger.warning(f"Paraphrased file not found: {paraphrased_path}, skipping train conversion")

    # -- ShareGPT (LLaMA-Factory) --
    if "sharegpt" in formats:
        if has_paraphrased:
            logger.info("=== Stage 5: ShareGPT Format Conversion (train) ===")
            train_path, info_path = convert_to_llamafactory(
                paraphrased_path=paraphrased_path,
                output_dir=args.output_dir,
            )
            paths.extend([train_path, info_path])

        eval_splits = [s for s in args.splits if s != "train"]
        if eval_splits:
            logger.info(f"=== Stage 5: ShareGPT Eval Conversion ({', '.join(eval_splits)}) ===")
            paths.extend(convert_eval_splits(output_dir=args.output_dir, splits=eval_splits))

    # -- CineMem flat JSONL --
    if "jsonl" in formats and has_paraphrased:
        logger.info("=== Stage 5: CineMem JSONL Conversion (train) ===")
        out = convert_to_jsonl_file(
            input_path=paraphrased_path,
            output_path=args.output_dir / "cardiac_vqa_train.jsonl",
        )
        paths.append(out)

    # -- VisMem JSONL --
    if "vismem" in formats and has_paraphrased:
        logger.info("=== Stage 5: VisMem JSONL Conversion (train) ===")
        out = convert_to_vismem_file(
            input_path=paraphrased_path,
            output_path=args.output_dir / "cardiac_vqa_train_vismem.jsonl",
            image_root=args.image_root,
        )
        paths.append(out)

    return paths


def run_push_to_hub(args: argparse.Namespace) -> None:
    """Push the final VQA dataset to HuggingFace Hub."""
    from cardio.data.vqa.convert import push_vqa_to_hub

    username = args.hf_username
    if not username:
        msg = "HuggingFace username required for push-to-hub. Set --hf-username."
        raise ValueError(msg)

    logger.info("=== Pushing VQA dataset to HuggingFace Hub ===")
    push_vqa_to_hub(
        output_dir=args.output_dir,
        repo_id=f"{username}/cardiac_cine_vqa",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the VQA generation pipeline."""
    load_dotenv()
    args = parse_args(argv)
    logger.info(f"Source: {args.source}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Format(s): {args.format}")

    dataset_dirs = _resolve_dataset_dirs(args)

    if args.stage:
        logger.info(f"Running single stage: {args.stage}")
        if args.stage == "ground_truth":
            run_ground_truth(args, dataset_dirs)
        elif args.stage == "reasoning":
            run_reasoning(args)
        elif args.stage == "validate":
            run_validate(args)
        elif args.stage == "paraphrase":
            run_paraphrase(args)
        elif args.stage == "convert":
            run_convert(args)
    else:
        logger.info("Running full pipeline")

        run_ground_truth(args, dataset_dirs)

        if "train" in args.splits:
            run_reasoning(args)
            run_validate(args)
            run_paraphrase(args)

        run_convert(args)

    if args.push_to_hub:
        run_push_to_hub(args)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
