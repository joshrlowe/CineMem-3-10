"""Script to preprocess kaggle DICOM data."""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813

from cardio.data.constants import UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE
from cardio.data.io.dicom import load_dicom_folder
from cardio.data.io.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_4d,
    crop_xy_4d,
    get_origin_for_crop,
    get_sax_center,
    resample_spacing_4d,
)
from cardio.utils.logging import get_logger
from cardio.vision.metric import ejection_fraction

KAGGLE_SPACING = (1.0, 1.0, 10.0)
KAGGLE_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
KAGGLE_LAX_SLICE_SIZE = UKB_LAX_SLICE_SIZE

logger = get_logger(__name__)

PIDS_TO_SKIP = [
    761,  # all black images
]


def find_longest_consecutive_subseq_with_same_values(
    lst: list[float | np.ndarray] | np.ndarray,
) -> tuple[int, int]:
    """Find longest consecutive subsequence with same values.

    Args:
        lst: list of values.

    Returns:
        start index and length of the longest consecutive subsequence.
    """
    best_n = 0
    n = 0
    best_start = -1
    start = -1
    for i, x in enumerate(lst):
        if i > 0 and np.all(x == lst[i - 1]):
            n += 1
        else:
            n = 1
            start = i
        if n > best_n:
            best_n = n
            best_start = start
    return best_start, best_n


def filter_sax_images(sax_images: list[sitk.Image], decimals: int) -> list[sitk.Image]:
    """Filter SAX images.

    Need to find consecutive slices such that
    the difference between the origins are consistent, this equals the spacing at z-axis,
    then the spacing should be the same,
    then need to align the directions, select the most comment direction.

    Args:
        sax_images: list of SAX images, GetSize() = (x, y, t).
        decimals: number of decimals to round the values.

    Returns:
        list of filtered SAX images, GetSize() = (x, y, t).
    """
    # check SAX image sizes
    sax_image_sizes = np.array([sax_image.GetSize() for sax_image in sax_images])
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_image_sizes)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX pixel spacing
    # (n_slices, 3)
    sax_pixel_spacings = np.round(np.array([image.GetSpacing() for image in sax_images]), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_pixel_spacings)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX slice direction
    # (n_slices, 9)
    sax_directions = np.round(np.array([image.GetDirection() for image in sax_images]), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_directions)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX slice spacing
    # (n_slices, 3)
    sax_origins = np.array([image.GetOrigin() for image in sax_images])
    # (n_slices-1,)
    sax_slice_spacings = np.round(np.linalg.norm(np.diff(sax_origins, axis=0), axis=-1), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_slice_spacings)
    sax_images = sax_images[start_index : start_index + slice_len + 1]  # +1 as the seq is on difference

    return sax_images


def _get_sax_geometric_center(sax_image: sitk.Image) -> np.ndarray:
    """Compute the geometric centre of a SAX image in world coordinates.

    Used as a fallback crop centre when LAX 2C/4C views are unavailable.

    Args:
        sax_image: SAX image, GetSize() = (x, y, z, t).

    Returns:
        centre coordinates in real space, shape (3,).
    """
    size = np.array(sax_image.GetSize()[:3], dtype=float)
    center_index = (size - 1.0) / 2.0
    spacing = np.array(sax_image.GetSpacing()[:3])
    origin = np.array(sax_image.GetOrigin()[:3])
    rot = np.array(sax_image.GetDirection()).reshape((4, 4))[:3, :3]
    return rot @ (center_index * spacing) + origin


def process_study(  # pylint:disable=too-many-statements
    study_dir: Path,
    pid: str,
    out_dir: Path,
    spacing: tuple[float, ...] = KAGGLE_SPACING,
    lax_slice_size: tuple[int, int] = KAGGLE_LAX_SLICE_SIZE,
    sax_slice_size: tuple[int, int] = KAGGLE_SAX_SLICE_SIZE,
) -> dict[str, int]:
    """Process DICOM in study and save nifti files.

    LAX 2-chamber and 4-chamber views are optional.  When both are present the
    crop centre is computed from the 2C/4C/SAX plane intersection; otherwise the
    geometric centre of the SAX stack is used as a fallback.

    Args:
        study_dir: directory of the study.
        pid: unique id of the study.
        out_dir: output directory.
        spacing: target spacing.
        lax_slice_size: slice size for cropping LAX images.
        sax_slice_size: slice size for cropping SAX images.

    Returns:
        dictionary of metadata, including pid, the number of slices and number of time frames.
    """
    # --- load optional 2C / 4C images ----------------------------------------
    dir_2c_matches = list(study_dir.glob("2ch_*"))
    dir_4c_matches = list(study_dir.glob("4ch_*"))

    lax_2c_image = None
    lax_4c_image = None

    if dir_2c_matches:
        lax_2c_image = load_dicom_folder([dir_2c_matches[0]], study_dir / "lax_2c.nii.gz")
        lax_2c_image = resample_spacing_4d(
            image=lax_2c_image,
            is_label=False,
            target_spacing=(*spacing[:2], lax_2c_image.GetSpacing()[-2]),
        )

    if dir_4c_matches:
        lax_4c_image = load_dicom_folder([dir_4c_matches[0]], study_dir / "lax_4c.nii.gz")
        lax_4c_image = resample_spacing_4d(
            image=lax_4c_image,
            is_label=False,
            target_spacing=(*spacing[:2], lax_4c_image.GetSpacing()[-2]),
        )

    has_2c = lax_2c_image is not None
    has_4c = lax_4c_image is not None

    if not has_2c:
        logger.warning(f"Study {pid}: no 2ch view found, skipping LAX 2C processing.")
    if not has_4c:
        logger.warning(f"Study {pid}: no 4ch view found, skipping LAX 4C processing.")

    # --- load SAX images (required) ------------------------------------------
    sax_dirs = sorted(study_dir.glob("sax_*"), key=lambda x: int(x.name.split("sax_")[1]))
    if not sax_dirs:
        raise ValueError(f"No sax_* directories found for study {pid}.")
    sax_image = load_dicom_folder(sax_dirs, study_dir / "sax.nii.gz")

    orig_sax_spacing = sax_image.GetSpacing()[:3]  # (x, y, z)
    sax_image = resample_spacing_4d(
        image=sax_image,
        is_label=False,
        target_spacing=spacing,
    )

    # --- compute crop centre --------------------------------------------------
    sax_center = None
    if has_2c and has_4c:
        sax_center = get_sax_center(
            sax_image=sax_image,
            lax_2c_image=lax_2c_image,
            lax_4c_image=lax_4c_image,
        )
        if sax_center is None:
            logger.warning(
                f"Study {pid}: 2C/4C/SAX plane intersection failed, "
                "falling back to SAX geometric centre."
            )

    if sax_center is None:
        sax_center = _get_sax_geometric_center(sax_image)

    # --- crop -----------------------------------------------------------------
    # Crop LAX 2C
    if has_2c:
        lax_2c_origin_indices = get_origin_for_crop(
            center=sax_center, image=lax_2c_image, slice_size=lax_slice_size,
        )
        lax_2c_image = crop_xy_4d(
            image=lax_2c_image, origin_indices=lax_2c_origin_indices, slice_size=lax_slice_size,
        )
        lax_2c_image = clip_and_normalise_intensity_4d(lax_2c_image, intensity_range=None)

    # Crop LAX 4C
    if has_4c:
        lax_4c_origin_indices = get_origin_for_crop(
            center=sax_center, image=lax_4c_image, slice_size=lax_slice_size,
        )
        lax_4c_image = crop_xy_4d(
            image=lax_4c_image, origin_indices=lax_4c_origin_indices, slice_size=lax_slice_size,
        )
        lax_4c_image = clip_and_normalise_intensity_4d(lax_4c_image, intensity_range=None)

    # Crop SAX
    sax_origin_indices = get_origin_for_crop(
        center=sax_center, image=sax_image, slice_size=sax_slice_size,
    )
    sax_image = crop_xy_4d(
        image=sax_image, origin_indices=sax_origin_indices, slice_size=sax_slice_size,
    )
    sax_image = clip_and_normalise_intensity_4d(sax_image, intensity_range=None)

    # --- save -----------------------------------------------------------------
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_2c:
        sitk.WriteImage(
            image=cast_to_uint8(lax_2c_image),
            fileName=out_dir / f"{pid}_lax_2c_t.nii.gz",
            useCompression=True,
        )
    if has_4c:
        sitk.WriteImage(
            image=cast_to_uint8(lax_4c_image),
            fileName=out_dir / f"{pid}_lax_4c_t.nii.gz",
            useCompression=True,
        )
    sitk.WriteImage(
        image=cast_to_uint8(sax_image),
        fileName=out_dir / f"{pid}_sax_t.nii.gz",
        useCompression=True,
    )

    # --- metadata -------------------------------------------------------------
    n_frames_parts = [sax_image.GetSize()[-1]]
    if has_2c:
        n_frames_parts.append(lax_2c_image.GetSize()[-1])
    if has_4c:
        n_frames_parts.append(lax_4c_image.GetSize()[-1])

    return {
        "pid": int(pid),
        "has_2c": has_2c,
        "has_4c": has_4c,
        "n_slices": sax_image.GetSize()[2],
        "n_frames": min(n_frames_parts),
        "original_sax_spacing_x": orig_sax_spacing[0],
        "original_sax_spacing_y": orig_sax_spacing[1],
        "original_sax_spacing_z": orig_sax_spacing[2],
    }


def try_process_study(study_dir: Path, pid: str, out_dir: Path) -> dict[str, int]:
    """Try to process a study and log error if failed.

    Args:
        study_dir: directory of the study.
        pid: unique id of the study.
        out_dir: output directory.

    Returns:
        dictionary of metadata, including pid, the number of slices and number of time frames.
    """
    try:
        return process_study(study_dir, pid, out_dir)
    except Exception:  # pylint: disable=broad-except
        logger.exception(f"Failed to process {pid} for {study_dir}.")
    return {}


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of data.",
        default="second-annual-data-science-bowl",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder saving output files.",
        default="processed",
    )
    parser.add_argument(
        "--max_n_cpus",
        type=int,
        help="Maximum number of cpus to use.",
        default=4,
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_cpus = min(cpu_count(), args.max_n_cpus)

    for split in ["train", "validate", "test"]:
        out_split = "val" if split == "validate" else split
        logger.info(f"Processing {split} split.")
        split_dir = args.data_dir / split / split
        study_dirs = list(split_dir.glob("*/study"))
        study_dirs = [study_dir for study_dir in study_dirs if int(study_dir.parent.name) not in PIDS_TO_SKIP]

        with Pool(n_cpus) as p:
            results = p.starmap_async(
                try_process_study,
                [
                    (
                        study_dir,
                        study_dir.parent.name,
                        args.out_dir / out_split,
                    )
                    for study_dir in study_dirs
                ],
            )
            data = [x for x in results.get() if len(x) > 0]

        # merge label and save metadata
        if split == "test":
            label_df = pd.read_csv(args.data_dir / "solution.csv")
            label_df["phase"] = label_df["Id"].apply(lambda x: x.split("_")[1])
            label_df["Id"] = label_df["Id"].apply(lambda x: int(x.split("_")[0]))
            label_df = label_df.pivot_table(index="Id", columns="phase", values="Volume").reset_index()
        else:
            label_df = pd.read_csv(args.data_dir / f"{split}.csv")
        label_df = label_df.rename(
            columns={
                "Id": "pid",
                "Systole": "systole_volume",
                "Diastole": "diastole_volume",
            },
            errors="raise",
        )
        label_df["ef"] = ejection_fraction(edv=label_df["diastole_volume"], esv=label_df["systole_volume"])
        meta_df = pd.DataFrame(data).sort_values("pid")
        meta_df = meta_df.merge(label_df, on="pid", how="left")
        meta_df_path = args.out_dir / f"{out_split}_metadata.csv"
        meta_df.to_csv(meta_df_path, index=False)
        logger.info(f"Saved metadata to {meta_df_path}.")


if __name__ == "__main__":
    main()
