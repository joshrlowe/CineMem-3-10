"""Stage 3: Validate GPT-4o reasoning outputs against ground truth."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from cardio.data.vqa.prompts import SYSTEM_PROMPT_REASONING, build_reasoning_prompt
from cardio.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_TOLERANCE = 0.15
BBOX_TOLERANCE = 3
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

ML_PER_M2_PATTERN = re.compile(r"\d+\.?\d*\s*mL/m[²2]", re.IGNORECASE)


def _parse_llm_response(raw: str | None) -> dict | None:
    """Parse the raw LLM response as JSON.

    Args:
        raw: raw string from GPT-4o.

    Returns:
        Parsed dict, or None if parsing fails.
    """
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _check_correct_option(parsed: dict, expected: str) -> str | None:
    """Verify the correct_option matches ground truth.

    Returns:
        Error message string if mismatch, None if ok.
    """
    got = parsed.get("correct_option", "").strip()
    if got != expected:
        return f"correct_option mismatch: expected '{expected}', got '{got}'"
    return None


def _check_metrics_in_reasoning(reasoning: str, metrics: dict[str, float]) -> list[str]:
    """Check that any numeric values cited in reasoning match input metrics.

    Only checks values that appear to be metric references (numbers
    followed by mL or %).

    Returns:
        List of error message strings (empty if all ok).
    """
    errors = []
    for key, value in metrics.items():
        str_val = f"{value:.1f}"
        short_val = f"{value:.0f}" if value == int(value) else str_val

        if str_val in reasoning or short_val in reasoning:
            continue

        rounded_val = round(value, 1)
        pattern = re.compile(rf"{re.escape(key.replace('_', ' '))}.*?(\d+\.?\d*)")
        for match in pattern.finditer(reasoning.lower()):
            cited = float(match.group(1))
            if abs(cited - rounded_val) > rounded_val * METRIC_TOLERANCE:
                errors.append(
                    f"Metric {key}: ground truth is {rounded_val}, reasoning cites {cited}"
                )
    return errors


def _check_bbox_match(parsed: dict, gt_grounding: dict) -> str | None:
    """Check that the grounding bbox matches ground truth.

    Returns:
        Error message string if mismatch, None if ok.
    """
    parsed_grounding = parsed.get("grounding", {})
    parsed_bbox = parsed_grounding.get("bbox", [])
    gt_bbox = gt_grounding.get("bbox", [])

    if not gt_bbox or gt_bbox == [-1, -1, -1, -1]:
        return None

    if not parsed_bbox or len(parsed_bbox) != len(gt_bbox):
        return f"bbox format mismatch: expected {gt_bbox}, got {parsed_bbox}"

    for i, (p, g) in enumerate(zip(parsed_bbox, gt_bbox)):
        if abs(p - g) > BBOX_TOLERANCE:
            return f"bbox mismatch at index {i}: expected {g}, got {p}"

    return None


def _check_ml_per_m2(reasoning: str) -> str | None:
    """Check that the reasoning does not cite mL/m2 thresholds.

    Returns:
        Error message if found, None if clean.
    """
    if ML_PER_M2_PATTERN.search(reasoning):
        return "Reasoning cites mL/m2 thresholds against absolute mL values"
    return None


def _validate_single(row: dict) -> tuple[dict | None, list[str]]:
    """Run all validation checks on a single reasoning row.

    Args:
        row: a dict from reasoning_raw.jsonl.

    Returns:
        (parsed_response, list_of_errors). parsed_response is None on parse failure.
    """
    errors = []

    if row.get("llm_status") != "success":
        return None, ["LLM call failed"]

    parsed = _parse_llm_response(row.get("llm_response_raw"))
    if parsed is None:
        return None, ["Failed to parse LLM response as JSON"]

    opt_err = _check_correct_option(parsed, row["correct_option"])
    if opt_err:
        errors.append(opt_err)

    reasoning_text = parsed.get("reasoning", "")

    metric_errs = _check_metrics_in_reasoning(reasoning_text, row.get("metrics", {}))
    errors.extend(metric_errs)

    gt_grounding = row.get("grounding_gt", {})
    if "structures" not in gt_grounding:
        bbox_err = _check_bbox_match(parsed, gt_grounding)
        if bbox_err:
            errors.append(bbox_err)

    ml_err = _check_ml_per_m2(reasoning_text)
    if ml_err:
        errors.append(ml_err)

    return parsed, errors


def _retry_with_feedback(
    client,
    row: dict,
    errors: list[str],
    model: str,
) -> tuple[dict | None, list[str]]:
    """Re-prompt GPT-4o with error feedback for failed validation.

    Args:
        client: openai.OpenAI client instance.
        row: the original reasoning row.
        errors: list of error strings from the first validation.
        model: model identifier.

    Returns:
        (parsed_response, remaining_errors) after retry.
    """
    error_feedback = "\n".join(f"- {e}" for e in errors)
    user_prompt = build_reasoning_prompt(
        category=row["category"],
        question_text=row["question_text"],
        options=row["options"],
        correct_option=row["correct_option"],
        metrics=row.get("metrics", {}),
        grounding=row.get("grounding_gt", {}),
        reasoning_template="",
    )
    user_prompt += f"\n\nYour previous response had these errors:\n{error_feedback}\nPlease correct them."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            row_copy = dict(row)
            row_copy["llm_response_raw"] = raw
            row_copy["llm_status"] = "success"
            parsed, new_errors = _validate_single(row_copy)
            if not new_errors:
                return parsed, []
            errors = new_errors
        except Exception:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.exception(f"Retry API call failed (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait:.1f}s")
            time.sleep(wait)

    return None, errors


def validate_reasoning(
    reasoning_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> tuple[Path, Path]:
    """Validate reasoning outputs and retry failures.

    Args:
        reasoning_path: path to reasoning_raw.jsonl.
        output_dir: directory for output files.
        api_key: OpenAI API key (required for retries).
        model: model identifier.

    Returns:
        Tuple of (canonical_vqa.jsonl path, validation_failures.jsonl path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_vqa.jsonl"
    failures_path = output_dir / "validation_failures.jsonl"

    client = None
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    rows = []
    with open(reasoning_path) as f:
        for line in f:
            rows.append(json.loads(line))

    valid_count = 0
    retry_count = 0
    fail_count = 0

    with open(canonical_path, "w") as f_valid, open(failures_path, "w") as f_fail:
        for row in tqdm(rows, desc="Validating"):
            parsed, errors = _validate_single(row)

            if not errors and parsed is not None:
                canonical_row = _build_canonical_row(row, parsed)
                f_valid.write(json.dumps(canonical_row) + "\n")
                valid_count += 1
                continue

            if client is not None and errors:
                retry_count += 1
                parsed, errors = _retry_with_feedback(client, row, errors, model)
                if not errors and parsed is not None:
                    canonical_row = _build_canonical_row(row, parsed)
                    f_valid.write(json.dumps(canonical_row) + "\n")
                    valid_count += 1
                    continue

            failure_row = {
                "study_id": row.get("study_id"),
                "category": row.get("category"),
                "errors": errors,
                "raw_response": row.get("llm_response_raw"),
            }
            f_fail.write(json.dumps(failure_row) + "\n")
            fail_count += 1

    logger.info(
        f"Validation complete: {valid_count} valid, {retry_count} retried, "
        f"{fail_count} permanently failed"
    )
    return canonical_path, failures_path


def _build_canonical_row(row: dict, parsed: dict) -> dict:
    """Build a canonical VQA row from the original row and parsed LLM response.

    Args:
        row: original reasoning_raw row.
        parsed: parsed LLM JSON response.

    Returns:
        Canonical VQA row dict.
    """
    return {
        "study_id": row["study_id"],
        "dataset": row["dataset"],
        "image": row["image"],
        "category": row["category"],
        "question_text": row["question_text"],
        "options": row["options"],
        "correct_option": row["correct_option"],
        "reasoning": parsed.get("reasoning", ""),
        "grounding": row.get("grounding_gt", {}),
        "metrics": row.get("metrics", {}),
        "pathology": row.get("pathology", ""),
    }
