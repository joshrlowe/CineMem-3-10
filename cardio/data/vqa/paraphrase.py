"""Stage 4: Generate paraphrased question variants via GPT-4o."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from cardio.data.vqa.prompts import SYSTEM_PROMPT_PARAPHRASE, build_paraphrase_prompt
from cardio.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
NUM_PARAPHRASES = 3


def _call_paraphrase(
    client,
    *,
    question_text: str,
    options: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> list[str] | None:
    """Call GPT-4o to paraphrase a question.

    Args:
        client: openai.OpenAI client instance.
        question_text: original question stem.
        options: list of option dicts.
        model: model identifier.

    Returns:
        List of 3 paraphrased question strings, or None on failure.
    """
    user_prompt = build_paraphrase_prompt(question_text=question_text, options=options)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_PARAPHRASE},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()

            # Parse the JSON array
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            paraphrases = json.loads(raw)
            if isinstance(paraphrases, list) and len(paraphrases) == NUM_PARAPHRASES:
                if all(isinstance(p, str) and len(p) > 10 for p in paraphrases):
                    return paraphrases

            logger.warning(f"Invalid paraphrase format, retrying (attempt {attempt + 1})")
        except Exception:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.exception(f"Paraphrase API call failed (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait:.1f}s")
            time.sleep(wait)

    return None


def generate_paraphrases(
    canonical_path: Path,
    output_dir: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Generate paraphrased question variants for each canonical VQA row.

    For each row, produces 3 additional rows with varied question text
    but identical answers, reasoning, grounding, and options.

    Args:
        canonical_path: path to canonical_vqa.jsonl.
        output_dir: directory for output.
        api_key: OpenAI API key.
        model: model identifier.

    Returns:
        Path to canonical_vqa_paraphrased.jsonl.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "canonical_vqa_paraphrased.jsonl"

    # Load existing completions for resume support
    existing_keys: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = f"{obj['study_id']}_{obj['category']}_{obj.get('variant', 'original')}"
                    existing_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resuming: {len(existing_keys)} rows already in output")

    # Load canonical rows
    rows = []
    with open(canonical_path) as f:
        for line in f:
            rows.append(json.loads(line))

    total_written = 0
    para_failed = 0

    with open(out_path, "a") as f_out:
        for row in tqdm(rows, desc="Paraphrasing"):
            original_key = f"{row['study_id']}_{row['category']}_original"

            # Write original row if not already present
            if original_key not in existing_keys:
                original_row = dict(row)
                original_row["variant"] = "original"
                f_out.write(json.dumps(original_row) + "\n")
                total_written += 1

            # Check if paraphrases already exist
            para_keys = [
                f"{row['study_id']}_{row['category']}_paraphrase_{i}"
                for i in range(NUM_PARAPHRASES)
            ]
            if all(k in existing_keys for k in para_keys):
                continue

            paraphrases = _call_paraphrase(
                client,
                question_text=row["question_text"],
                options=row["options"],
                model=model,
            )

            if paraphrases is None:
                para_failed += 1
                logger.warning(
                    f"Failed to paraphrase {row['study_id']}_{row['category']}"
                )
                continue

            for i, para_text in enumerate(paraphrases):
                para_key = f"{row['study_id']}_{row['category']}_paraphrase_{i}"
                if para_key in existing_keys:
                    continue
                para_row = dict(row)
                para_row["question_text"] = para_text
                para_row["variant"] = f"paraphrase_{i}"
                f_out.write(json.dumps(para_row) + "\n")
                total_written += 1

    logger.info(
        f"Paraphrasing complete: {total_written} new rows written, "
        f"{para_failed} questions failed paraphrasing. Output: {out_path}"
    )
    return out_path
