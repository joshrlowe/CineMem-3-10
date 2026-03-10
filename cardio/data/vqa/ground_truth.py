"""Stage 1: Generate ground truth JSONL from preprocessed metadata and segmentation masks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
from PIL import Image
from tqdm import tqdm

import pandas as pd

from cardio.data.constants import LV_LABEL, MYO_LABEL, RV_LABEL
from cardio.data.io.sitk import get_binary_mask_bounding_box
from cardio.data.vqa import (
    ALL_DATASETS,
    CAT1_ANSWER_MAP,
    CAT1_GROUNDING_MAP,
    CAT1_OPTIONS,
    CAT1_QUESTION,
    CAT2_OPTIONS,
    CAT2_QUESTION,
    CAT3_LV_DILATED,
    CAT3_LV_QUESTION,
    CAT3_OPTIONS,
    CAT3_RV_DILATED,
    CAT3_RV_QUESTION,
    CAT4_ACDC_POOL,
    CAT4_GROUNDING_MAP,
    CAT4_LABEL_TO_DISPLAY,
    CAT4_MNMS2_POOL,
    CAT4_MNMS_POOL,
    CAT4_QUESTION,
    CAT5_ANSWER_MAP,
    CAT5_OPTIONS,
    CAT5_QUESTION,
    CAT6_OPTIONS,
    CAT6_QUESTION,
    CAT_ABNORMALITY,
    CAT_BIVENTRICULAR,
    CAT_DIAGNOSIS,
    CAT_DILATION_LV,
    CAT_DILATION_RV,
    CAT_ED_ES,
    CAT_LV_FUNCTION,
    DATASET_ACDC,
    DATASET_MNMS,
    DATASET_MNMS2,
    EXCLUDED_PATHOLOGIES,
    get_cat2_answer,
    get_cat6_answer,
)
from cardio.data.vqa.prompts import (
    CAT1_TEMPLATES,
    CAT2_TEMPLATES,
    CAT3_LV_TEMPLATES,
    CAT3_RV_TEMPLATES,
    CAT4_TEMPLATES,
    CAT5_TEMPLATES,
    CAT6_TEMPLATES,
)
from cardio.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_middle_slice(nifti_path: Path) -> tuple[np.ndarray, int]:
    """Load a 3D NIfTI and extract the middle slice along the z-axis.

    Args:
        nifti_path: path to the NIfTI file.

    Returns:
        2D numpy array (x, y) of the middle slice, and the slice index used.
    """
    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img)
    n_slices = arr.shape[0]
    slice_idx = n_slices // 2
    return arr[slice_idx], slice_idx


def _save_slice_as_png(
    slice_arr: np.ndarray,
    out_path: Path,
) -> None:
    """Save a 2D numpy array as a grayscale PNG.

    Handles both float (0-1) and integer value ranges by normalising
    the slice into the [0, 255] uint8 range.

    Args:
        slice_arr: 2D array (float or int).
        out_path: output file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = slice_arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-8:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(out_path)


def _compute_bbox_2d(mask_slice: np.ndarray, label: int) -> list[int]:
    """Compute a 2D bounding box for a given label on a single slice.

    The underlying ``get_binary_mask_bounding_box`` returns coordinates in
    array axis order ``[row/y, col/x]``. We swap to ``[x1, y1, x2, y2]``
    to match the JSON schema convention.

    Args:
        mask_slice: 2D segmentation mask.
        label: the integer label to extract bbox for.

    Returns:
        [x1, y1, x2, y2] list, or [-1, -1, -1, -1] if label not present.
    """
    binary = (mask_slice == label)
    bbox_min, bbox_max = get_binary_mask_bounding_box(binary)
    # bbox_min/max are [row_start, col_start] / [row_end, col_end]
    # Convert to [x1, y1, x2, y2] where x=col, y=row
    return [int(bbox_min[1]), int(bbox_min[0]), int(bbox_max[1]), int(bbox_max[0])]


def _build_cat4_options(
    pathology: str,
    dataset: str,
) -> list[dict[str, str]]:
    """Build Category 4 options: correct diagnosis + 3 distractors.

    Args:
        pathology: the patient's ground truth pathology label.
        dataset: the dataset name.

    Returns:
        List of 4 option dicts with id and text.
    """
    if dataset == DATASET_ACDC:
        pool = CAT4_ACDC_POOL
    elif dataset == DATASET_MNMS:
        pool = CAT4_MNMS_POOL
    else:
        pool = CAT4_MNMS2_POOL

    # Normalize M&Ms2 short labels to canonical pool names
    _ALIAS = {"RV": "ARV", "LV": "DLV"}
    normalized = _ALIAS.get(pathology, pathology)
    distractors = [p for p in pool if p != normalized and p != pathology]
    selected = random.sample(distractors, min(3, len(distractors)))

    all_options = [normalized] + selected
    random.shuffle(all_options)

    options = []
    for i, p in enumerate(all_options):
        letter = chr(ord("A") + i)
        options.append({"id": letter, "text": CAT4_LABEL_TO_DISPLAY.get(p, p)})

    correct_letter = next(
        o["id"] for o in options if o["text"] == CAT4_LABEL_TO_DISPLAY.get(normalized, normalized)
    )
    return options, correct_letter


def _format_template(template: str, metrics: dict[str, float]) -> str:
    """Fill a reasoning template with patient metric values.

    Args:
        template: string with {lv_edv}, {lv_ef}, etc. placeholders.
        metrics: dict of patient metrics.

    Returns:
        Formatted string.
    """
    return template.format(
        lv_edv=round(metrics.get("lv_edv", 0), 1),
        lv_esv=round(metrics.get("lv_esv", 0), 1),
        lv_ef=round(metrics.get("lv_ef", 0), 1),
        rv_edv=round(metrics.get("rv_edv", 0), 1),
        rv_esv=round(metrics.get("rv_esv", 0), 1),
        rv_ef=round(metrics.get("rv_ef", 0), 1),
    )


def _get_patient_metrics(row: pd.Series) -> dict[str, float]:
    """Extract metric dict from a metadata DataFrame row."""
    return {
        "lv_edv": float(row["lv_edv"]),
        "lv_esv": float(row["lv_esv"]),
        "lv_ef": float(row["lv_ef"]),
        "rv_edv": float(row["rv_edv"]),
        "rv_esv": float(row["rv_esv"]),
        "rv_ef": float(row["rv_ef"]),
    }


def _resolve_metadata_csv(data_root: Path, split: str) -> Path | None:
    """Find the metadata CSV for a given split.

    Checks ``{split}_metadata.csv`` first, then falls back to ``{split}.csv``.
    Returns ``None`` if neither exists.
    """
    for name in [f"{split}_metadata.csv", f"{split}.csv"]:
        p = data_root / name
        if p.exists():
            return p
    return None


def _process_split_for_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    split: str,
    output_dir: Path,
    seed: int,
    f,
) -> tuple[int, int]:
    """Process one split of one dataset, writing rows to *f*.

    Args:
        dataset_name: e.g. ``"acdc"``.
        data_root: root of the downloaded/local dataset (contains
            ``{split}_metadata.csv`` and ``{split}/{pid}/...``).
        split: ``"train"``, ``"val"``, or ``"test"``.
        output_dir: base output directory for images and JSONL.
        seed: random seed (already seeded globally, but documented).
        f: open file handle to write JSONL rows.

    Returns:
        ``(total_rows_written, flagged_empty_bbox_count)``
    """
    csv_path = _resolve_metadata_csv(data_root, split)
    if csv_path is None:
        logger.warning(
            f"Metadata not found for {dataset_name}/{split} in {data_root}, skipping"
        )
        return 0, 0

    df = pd.read_csv(csv_path)
    logger.info(f"[{dataset_name}/{split}] Loaded {len(df)} patients from {csv_path}")

    total_rows = 0
    flagged_empty = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name}/{split}"):
        pid = str(row["pid"])
        pathology = str(row["pathology"]).strip()

        if pathology in EXCLUDED_PATHOLOGIES:
            continue

        if pathology not in CAT1_ANSWER_MAP:
            logger.warning(f"Unknown pathology {pathology} for {pid}, skipping")
            continue

        metrics = _get_patient_metrics(row)
        patient_dir = data_root / split / pid
        study_id = f"{dataset_name}_{pid}"

        # --- Load ED image and mask, extract middle slice ---
        ed_img_path = patient_dir / f"{pid}_sax_ed.nii.gz"
        ed_gt_path = patient_dir / f"{pid}_sax_ed_gt.nii.gz"

        if not ed_img_path.exists() or not ed_gt_path.exists():
            logger.warning(f"Missing ED files for {pid}, skipping")
            continue

        ed_slice, slice_idx = _extract_middle_slice(ed_img_path)
        ed_mask_slice, _ = _extract_middle_slice(ed_gt_path)

        ed_png_rel = f"images/{dataset_name}/{pid}_sax_ed_slice{slice_idx}.png"
        ed_png_path = output_dir / ed_png_rel
        _save_slice_as_png(ed_slice, ed_png_path)

        bboxes = {
            LV_LABEL: _compute_bbox_2d(ed_mask_slice, LV_LABEL),
            MYO_LABEL: _compute_bbox_2d(ed_mask_slice, MYO_LABEL),
            RV_LABEL: _compute_bbox_2d(ed_mask_slice, RV_LABEL),
        }

        # --- Load ES image and mask for Category 6 ---
        es_img_path = patient_dir / f"{pid}_sax_es.nii.gz"
        es_gt_path = patient_dir / f"{pid}_sax_es_gt.nii.gz"
        es_png_rel = None
        es_lv_bbox = None

        if es_img_path.exists() and es_gt_path.exists():
            es_slice, es_slice_idx = _extract_middle_slice(es_img_path)
            es_mask_slice, _ = _extract_middle_slice(es_gt_path)

            es_png_rel = f"images/{dataset_name}/{pid}_sax_es_slice{es_slice_idx}.png"
            _save_slice_as_png(es_slice, output_dir / es_png_rel)
            es_lv_bbox = _compute_bbox_2d(es_mask_slice, LV_LABEL)

        # Common base fields injected into every row
        base = {"split": split}

        # ==============================================================
        # Generate rows for all 7 question types
        # ==============================================================

        # --- Category 1: Abnormality Localization ---
        cat1_answer = CAT1_ANSWER_MAP[pathology]
        cat1_struct, cat1_label = CAT1_GROUNDING_MAP[cat1_answer]
        cat1_bbox = bboxes.get(cat1_label, [-1, -1, -1, -1])

        if cat1_label != -1 and cat1_bbox == [-1, -1, -1, -1]:
            flagged_empty += 1

        cat1_template = _format_template(
            CAT1_TEMPLATES.get(pathology, ""),
            metrics,
        )

        row_cat1 = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_ABNORMALITY,
            "question_text": CAT1_QUESTION,
            "options": CAT1_OPTIONS,
            "correct_option": cat1_answer,
            "grounding": {"structure": cat1_struct, "bbox": cat1_bbox, "source_mask_label": cat1_label},
            "metrics": metrics,
            "reasoning_template": cat1_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat1) + "\n")
        total_rows += 1

        # --- Category 2: LV Systolic Function ---
        cat2_answer = get_cat2_answer(metrics["lv_ef"])
        cat2_template = _format_template(CAT2_TEMPLATES[cat2_answer], metrics)
        lv_bbox = bboxes[LV_LABEL]

        row_cat2 = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_LV_FUNCTION,
            "question_text": CAT2_QUESTION,
            "options": CAT2_OPTIONS,
            "correct_option": cat2_answer,
            "grounding": {"structure": "LV", "bbox": lv_bbox, "source_mask_label": LV_LABEL},
            "metrics": metrics,
            "reasoning_template": cat2_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat2) + "\n")
        total_rows += 1

        # --- Category 3: LV Dilation ---
        cat3lv_answer = CAT3_LV_DILATED[pathology]
        cat3lv_template = _format_template(CAT3_LV_TEMPLATES[cat3lv_answer], metrics)

        row_cat3lv = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_DILATION_LV,
            "question_text": CAT3_LV_QUESTION,
            "options": CAT3_OPTIONS,
            "correct_option": cat3lv_answer,
            "grounding": {"structure": "LV", "bbox": lv_bbox, "source_mask_label": LV_LABEL},
            "metrics": metrics,
            "reasoning_template": cat3lv_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat3lv) + "\n")
        total_rows += 1

        # --- Category 3: RV Dilation ---
        cat3rv_answer = CAT3_RV_DILATED[pathology]
        cat3rv_template = _format_template(CAT3_RV_TEMPLATES[cat3rv_answer], metrics)
        rv_bbox = bboxes[RV_LABEL]

        row_cat3rv = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_DILATION_RV,
            "question_text": CAT3_RV_QUESTION,
            "options": CAT3_OPTIONS,
            "correct_option": cat3rv_answer,
            "grounding": {"structure": "RV", "bbox": rv_bbox, "source_mask_label": RV_LABEL},
            "metrics": metrics,
            "reasoning_template": cat3rv_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat3rv) + "\n")
        total_rows += 1

        # --- Category 4: Diagnosis ---
        cat4_options, cat4_answer = _build_cat4_options(pathology, dataset_name)
        cat4_struct, cat4_label = CAT4_GROUNDING_MAP.get(
            pathology, ("NONE", -1)
        )
        cat4_bbox = bboxes.get(cat4_label, [-1, -1, -1, -1])
        cat4_template = _format_template(
            CAT4_TEMPLATES.get(pathology, ""),
            metrics,
        )

        row_cat4 = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_DIAGNOSIS,
            "question_text": CAT4_QUESTION,
            "options": cat4_options,
            "correct_option": cat4_answer,
            "grounding": {"structure": cat4_struct, "bbox": cat4_bbox, "source_mask_label": cat4_label},
            "metrics": metrics,
            "reasoning_template": cat4_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat4) + "\n")
        total_rows += 1

        # --- Category 5: Biventricular Comparison ---
        cat5_answer = CAT5_ANSWER_MAP[pathology]
        cat5_template = _format_template(CAT5_TEMPLATES.get(cat5_answer, ""), metrics)

        row_cat5 = {
            **base,
            "study_id": study_id,
            "dataset": dataset_name,
            "image": {"view": "SAX", "phase": "ED", "slice_idx": slice_idx, "path": ed_png_rel},
            "category": CAT_BIVENTRICULAR,
            "question_text": CAT5_QUESTION,
            "options": CAT5_OPTIONS,
            "correct_option": cat5_answer,
            "grounding": {
                "structures": [
                    {"structure": "LV", "bbox": lv_bbox, "source_mask_label": LV_LABEL},
                    {"structure": "RV", "bbox": rv_bbox, "source_mask_label": RV_LABEL},
                ],
            },
            "metrics": metrics,
            "reasoning_template": cat5_template,
            "pathology": pathology,
        }
        f.write(json.dumps(row_cat5) + "\n")
        total_rows += 1

        # --- Category 6: ED vs ES Phase Comparison ---
        if es_png_rel is not None:
            cat6_answer = get_cat6_answer(metrics["lv_ef"])
            cat6_template = _format_template(CAT6_TEMPLATES[cat6_answer], metrics)

            row_cat6 = {
                **base,
                "study_id": study_id,
                "dataset": dataset_name,
                "image": {
                    "view": "SAX",
                    "phase": "ED+ES",
                    "slice_idx": slice_idx,
                    "path_ed": ed_png_rel,
                    "path_es": es_png_rel,
                },
                "category": CAT_ED_ES,
                "question_text": CAT6_QUESTION,
                "options": CAT6_OPTIONS,
                "correct_option": cat6_answer,
                "grounding": {
                    "structures": [
                        {"structure": "LV_ED", "bbox": lv_bbox, "source_mask_label": LV_LABEL},
                        {"structure": "LV_ES", "bbox": es_lv_bbox, "source_mask_label": LV_LABEL},
                    ],
                },
                "metrics": metrics,
                "reasoning_template": cat6_template,
                "pathology": pathology,
            }
            f.write(json.dumps(row_cat6) + "\n")
            total_rows += 1

    return total_rows, flagged_empty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_ground_truth(
    output_dir: Path,
    dataset_dirs: dict[str, Path],
    splits: list[str] | None = None,
    seed: int = 42,
    *,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate ground-truth JSONL files for each requested split.

    Args:
        output_dir: directory where images/ and ground_truth_*.jsonl are written.
        dataset_dirs: mapping of dataset name to its local root directory
            (the directory that contains ``{split}_metadata.csv`` and
            ``{split}/{pid}/...``).  When using the HuggingFace source this
            comes from ``hf_loader.download_datasets()``.
        splits: list of splits to process (default: ``["train"]``).
        seed: random seed for reproducible distractor selection.
        data_dir: **deprecated** legacy local-mode path.  When given and
            *dataset_dirs* is empty, dataset dirs are resolved as
            ``data_dir/{name}/processed``.

    Returns:
        Dict mapping each split name to its output JSONL path.
    """
    if splits is None:
        splits = ["train"]

    # Backward-compat: derive dataset_dirs from legacy data_dir
    if not dataset_dirs and data_dir is not None:
        dataset_dirs = {
            DATASET_ACDC: data_dir / "acdc" / "processed",
            DATASET_MNMS: data_dir / "mnms" / "processed",
            DATASET_MNMS2: data_dir / "mnms2" / "processed",
        }

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    for split in splits:
        out_jsonl = output_dir / f"ground_truth_{split}.jsonl"
        total_rows = 0
        flagged_empty = 0

        with open(out_jsonl, "w") as f:
            for dataset_name, data_root in dataset_dirs.items():
                rows, empty = _process_split_for_dataset(
                    dataset_name=dataset_name,
                    data_root=data_root,
                    split=split,
                    output_dir=output_dir,
                    seed=seed,
                    f=f,
                )
                total_rows += rows
                flagged_empty += empty

        logger.info(f"[{split}] Ground truth complete: {total_rows} rows -> {out_jsonl}")
        if flagged_empty > 0:
            logger.warning(f"[{split}] {flagged_empty} rows have empty bboxes on the middle slice")

        output_paths[split] = out_jsonl

    return output_paths
