"""Stage 2: Generate clinical reasoning via GPT-4o API calls."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from cardio.data.vqa.prompts import (
    SYSTEM_PROMPT_REASONING,
    build_reasoning_prompt,
)
from cardio.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


def _call_gpt4o(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Make a single GPT-4o API call with retry logic.

    Args:
        client: openai.OpenAI client instance.
        system_prompt: the system message.
        user_prompt: the user message.
        model: model identifier.

    Returns:
        The raw response content string, or None if all retries failed.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.exception(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait:.1f}s")
            time.sleep(wait)

    return None


def generate_reasoning(
    ground_truth_path: Path,
    output_dir: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Generate reasoning for each ground truth row using GPT-4o.

    This function is resumable: it checks which study_id + category
    combinations already exist in the output file and skips them.

    Args:
        ground_truth_path: path to ground_truth.jsonl.
        output_dir: directory for output files.
        api_key: OpenAI API key.
        model: model identifier.

    Returns:
        Path to reasoning_raw.jsonl.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reasoning_raw.jsonl"

    # Load existing completions for resume support
    existing_keys: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = f"{obj['study_id']}_{obj['category']}"
                    existing_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resuming: {len(existing_keys)} rows already completed")

    # Load ground truth rows
    gt_rows = []
    with open(ground_truth_path) as f:
        for line in f:
            gt_rows.append(json.loads(line))

    pending = [r for r in gt_rows if f"{r['study_id']}_{r['category']}" not in existing_keys]
    logger.info(f"Processing {len(pending)} rows ({len(gt_rows) - len(pending)} already done)")

    success_count = 0
    fail_count = 0

    with open(out_path, "a") as f_out:
        for row in tqdm(pending, desc="GPT-4o reasoning"):
            user_prompt = build_reasoning_prompt(
                category=row["category"],
                question_text=row["question_text"],
                options=row["options"],
                correct_option=row["correct_option"],
                metrics=row["metrics"],
                grounding=row.get("grounding", {}),
                reasoning_template=row.get("reasoning_template", ""),
            )

            raw_response = _call_gpt4o(
                client,
                system_prompt=SYSTEM_PROMPT_REASONING,
                user_prompt=user_prompt,
                model=model,
            )

            output_row = {
                "study_id": row["study_id"],
                "dataset": row["dataset"],
                "image": row["image"],
                "category": row["category"],
                "question_text": row["question_text"],
                "options": row["options"],
                "correct_option": row["correct_option"],
                "grounding_gt": row.get("grounding", {}),
                "metrics": row["metrics"],
                "pathology": row.get("pathology", ""),
                "llm_response_raw": raw_response,
            }

            if raw_response is None:
                output_row["llm_status"] = "failed"
                fail_count += 1
            else:
                output_row["llm_status"] = "success"
                success_count += 1

            f_out.write(json.dumps(output_row) + "\n")

    logger.info(
        f"Reasoning generation complete: {success_count} succeeded, {fail_count} failed. "
        f"Output: {out_path}"
    )
    return out_path
