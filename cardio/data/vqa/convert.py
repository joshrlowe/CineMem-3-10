"""Unified format converters for VQA data.

Three public entry-point functions convert canonical VQA entries (dicts with
a ``category`` field) into downstream training formats:

- ``to_sharegpt`` — LLaMA-Factory ShareGPT (multi-turn JSON)
- ``to_jsonl``    — CineMem flat JSONL
- ``to_vismem``   — VisMem JSONL

All three read the ``category`` field directly from the structured entry,
replacing the fragile regex-based heuristics from the legacy bridge scripts
(``convert_cinema_to_cinemem.py`` and ``convert_to_vismem.py``).

File-level helpers (``convert_to_llamafactory``, ``convert_eval_splits``,
``push_vqa_to_hub``) wrap the per-entry converters with I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from cardio.data.vqa import CAT_ED_ES
from cardio.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_MESSAGE = (
    "You are a cardiac MRI interpretation assistant specialising in "
    "short-axis cine MRI analysis. You provide structured answers with "
    "clinical reasoning grounded in measurable cardiac metrics and spatial "
    "localisation of anatomical structures."
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_image_paths(row: dict) -> list[str]:
    """Extract image path(s) from a canonical VQA row.

    Category 6 (ED-vs-ES comparison) carries two images; all others one.
    """
    image = row.get("image", {})
    if row.get("category") == CAT_ED_ES:
        paths = []
        if "path_ed" in image:
            paths.append(image["path_ed"])
        if "path_es" in image:
            paths.append(image["path_es"])
        return paths
    path = image.get("path", "") if isinstance(image, dict) else (image or "")
    return [path] if path else []


def _build_user_content(row: dict) -> str:
    """Build the user content string for a ShareGPT conversation turn."""
    options_str = "\n".join(f"({o['id']}) {o['text']}" for o in row.get("options", []))
    question = row.get("question_text", "")

    if row.get("category") == CAT_ED_ES:
        return f"<image>\n<image>\n{question}\n{options_str}"
    return f"<image>\n{question}\n{options_str}"


def _build_assistant_content(row: dict) -> str:
    """Build the assistant content as a structured JSON string."""
    answer_obj = {
        "correct_option": row.get("correct_option", ""),
        "reasoning": row.get("reasoning", ""),
        "grounding": row.get("grounding", {}),
    }
    return json.dumps(answer_obj, ensure_ascii=False)


def _build_assistant_content_eval(row: dict) -> str:
    """Build assistant content for eval rows (reasoning template, no LLM reasoning)."""
    answer_obj = {
        "correct_option": row.get("correct_option", ""),
        "reasoning": row.get("reasoning_template", ""),
        "grounding": row.get("grounding", {}),
    }
    return json.dumps(answer_obj, ensure_ascii=False)


def _derive_entry_id(row: dict, idx: int) -> str:
    """Derive a deterministic ID from study_id + category, with a fallback."""
    study_id = row.get("study_id", "")
    category = row.get("category", "unknown")
    variant = row.get("variant", "")
    if study_id:
        base = f"{study_id}_{category}"
        return f"{base}_{variant}" if variant else base
    return f"cardiac_{idx:05d}"


# ===================================================================
# Public per-entry converters
# ===================================================================


def to_sharegpt(entries: list[dict]) -> list[dict]:
    """Convert canonical VQA entries to LLaMA-Factory ShareGPT format.

    Each output dict has ``messages`` (system / user / assistant) and
    ``images`` (list of relative paths).
    """
    results: list[dict] = []
    for row in entries:
        results.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": _build_user_content(row)},
                {"role": "assistant", "content": _build_assistant_content(row)},
            ],
            "images": _get_image_paths(row),
        })
    return results


def to_jsonl(entries: list[dict]) -> list[dict]:
    """Convert canonical VQA entries to CineMem flat JSONL format.

    Each output dict has ``id``, ``prompt``, ``answer``, ``category``,
    ``dataset``, and optionally ``image`` and ``meta``.

    The ``category`` field is read directly from the entry — no regex
    heuristics on the question text.
    """
    results: list[dict] = []
    for idx, row in enumerate(entries):
        images = _get_image_paths(row)
        image_field: str | list[str] | None = None
        if len(images) == 1:
            image_field = images[0]
        elif len(images) > 1:
            image_field = images

        options_str = "\n".join(f"({o['id']}) {o['text']}" for o in row.get("options", []))
        prompt = row.get("question_text", "")
        if options_str:
            prompt = f"{prompt}\n{options_str}"

        record: dict = {
            "id": _derive_entry_id(row, idx),
            "prompt": prompt,
            "answer": _build_assistant_content(row),
            "category": row.get("category", "unknown"),
            "dataset": row.get("dataset", ""),
        }
        if image_field is not None:
            record["image"] = image_field
        record["meta"] = {
            "study_id": row.get("study_id", ""),
            "pathology": row.get("pathology", ""),
        }
        results.append(record)
    return results


def to_vismem(entries: list[dict], *, image_root: str | None = None) -> list[dict]:
    """Convert canonical VQA entries to VisMem JSONL format.

    Each output dict has ``id``, ``prompt``, ``answer``, ``category``, and
    optionally ``image`` (single path or list of paths).

    The ``category`` field is read directly from the entry — no regex
    heuristics on the question text.

    Args:
        entries: canonical VQA entry dicts.
        image_root: if given, image paths are resolved relative to this
            directory and stored as absolute paths.
    """
    results: list[dict] = []
    for idx, row in enumerate(entries):
        raw_images = _get_image_paths(row)
        if image_root:
            images = [str(Path(image_root).resolve() / p) for p in raw_images]
        else:
            images = raw_images

        image_field: str | list[str] | None = None
        if len(images) == 1:
            image_field = images[0]
        elif len(images) > 1:
            image_field = images

        user_content = _build_user_content(row)
        prompt = f"{SYSTEM_MESSAGE}\n\n{user_content}" if SYSTEM_MESSAGE else user_content

        record: dict = {
            "id": _derive_entry_id(row, idx),
            "prompt": prompt,
            "answer": _build_assistant_content(row),
        }
        if image_field is not None:
            record["image"] = image_field
        record["category"] = row.get("category", "unknown")
        results.append(record)
    return results


# ===================================================================
# File-level operations (wrap per-entry converters with I/O)
# ===================================================================


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def convert_to_llamafactory(
    paraphrased_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Convert paraphrased JSONL to LLaMA-Factory ShareGPT JSON.

    Returns:
        Tuple of (training JSON path, dataset_info.json path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "cardiac_vqa_train.json"
    info_path = output_dir / "dataset_info.json"

    rows = _load_jsonl(paraphrased_path)
    logger.info(f"Converting {len(rows)} rows to LLaMA-Factory format")

    sharegpt_data = to_sharegpt(rows)

    with open(train_path, "w") as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "cardiac_vqa": {
            "file_name": str(train_path.name),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Conversion complete: {len(sharegpt_data)} entries -> {train_path}")
    return train_path, info_path


def convert_eval_splits(
    output_dir: Path,
    splits: list[str],
) -> list[Path]:
    """Convert ground-truth JSONL for val/test splits into eval ShareGPT JSON.

    Eval data is NOT paraphrased — each patient/category pair produces one
    conversation entry.  The assistant response uses the reasoning template.
    """
    output_paths: list[Path] = []

    for split in splits:
        gt_path = output_dir / f"ground_truth_{split}.jsonl"
        if not gt_path.exists():
            logger.warning(f"Ground truth not found for split '{split}': {gt_path}, skipping")
            continue

        out_path = output_dir / f"cardiac_vqa_{split}.json"
        rows = _load_jsonl(gt_path)

        logger.info(f"[{split}] Converting {len(rows)} eval rows to ShareGPT")

        sharegpt_data: list[dict] = []
        for row in tqdm(rows, desc=f"Eval {split}"):
            sharegpt_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": _build_user_content(row)},
                    {"role": "assistant", "content": _build_assistant_content_eval(row)},
                ],
                "images": _get_image_paths(row),
            })

        with open(out_path, "w") as f:
            json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[{split}] Eval conversion complete: {len(sharegpt_data)} entries -> {out_path}")
        output_paths.append(out_path)

    return output_paths


def convert_to_jsonl_file(
    input_path: Path,
    output_path: Path,
) -> Path:
    """Convert canonical/paraphrased JSONL to CineMem flat JSONL."""
    rows = _load_jsonl(input_path)
    records = to_jsonl(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"JSONL conversion: {len(records)} entries -> {output_path}")
    return output_path


def convert_to_vismem_file(
    input_path: Path,
    output_path: Path,
    *,
    image_root: str | None = None,
) -> Path:
    """Convert canonical/paraphrased JSONL to VisMem JSONL."""
    rows = _load_jsonl(input_path)
    records = to_vismem(rows, image_root=image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"VisMem conversion: {len(records)} entries -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------


def push_vqa_to_hub(
    output_dir: Path,
    repo_id: str,
) -> None:
    """Upload the generated VQA dataset to HuggingFace Hub.

    Uploads all JSON/JSONL output files plus the ``images/`` directory.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    logger.info(f"Uploading VQA dataset to {repo_id} ...")

    readme_path = output_dir / "README.md"
    if readme_path.is_file():
        logger.info("  Uploading README.md ...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    for pattern in ["cardiac_vqa_*.json", "cardiac_vqa_*.jsonl", "dataset_info.json"]:
        for fp in sorted(output_dir.glob(pattern)):
            logger.info(f"  Uploading {fp.name} ...")
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=repo_id,
                repo_type="dataset",
            )

    images_dir = output_dir / "images"
    if images_dir.is_dir():
        logger.info("  Uploading images/ folder (this may take a while) ...")
        api.upload_folder(
            folder_path=str(images_dir),
            path_in_repo="images",
            repo_id=repo_id,
            repo_type="dataset",
        )

    logger.info(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
