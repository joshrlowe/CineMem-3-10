import argparse
from pathlib import Path
import pandas as pd
import sys
from huggingface_hub import HfApi


def upload_readme(repo_id: str, readme_path: Path) -> None:
    if not readme_path.exists():
        print(f"  ⚠ README not found at {readme_path}, skipping upload.")
        return

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  ✓ README.md uploaded")




def _rel(path: Path, base: Path) -> str | None:
    """Return path relative to base, or None if path doesn't exist."""
    if path.exists():
        return str(path.relative_to(base))
    return None


def build_acdc_split(split_dir, metadata_csv, base_dir=None):
    split_dir = Path(split_dir)
    base = Path(base_dir) if base_dir else split_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(split_dir.iterdir()):
        if not pid_dir.is_dir():
            continue

        pid = pid_dir.name
        sax_ed = pid_dir / f"{pid}_sax_ed.nii.gz"
        sax_ed_gt = pid_dir / f"{pid}_sax_ed_gt.nii.gz"
        sax_es = pid_dir / f"{pid}_sax_es.nii.gz"
        sax_es_gt = pid_dir / f"{pid}_sax_es_gt.nii.gz"
        sax_t = pid_dir / f"{pid}_sax_t.nii.gz"

        if not (sax_ed.exists() and sax_es.exists() and sax_t.exists()):
            continue

        row = {
            "pid": pid,
            "sax_ed": _rel(sax_ed, base),
            "sax_ed_gt": _rel(sax_ed_gt, base),
            "sax_es": _rel(sax_es, base),
            "sax_es_gt": _rel(sax_es_gt, base),
            "sax_t": _rel(sax_t, base),
        }

        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)

        rows.append(row)

    return pd.DataFrame(rows)


def build_emidec_ds(data_dir, metadata_csv=None, base_dir=None):
    """Build EMIDEC split. All patients in data_dir; CSV selects the split."""
    data_dir = Path(data_dir)
    base = Path(base_dir) if base_dir else data_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        split_pids = set(meta_df["pid"].tolist())
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        split_pids = None
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(data_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        if split_pids is not None and pid not in split_pids:
            continue

        image_path = pid_dir / f"{pid}.nii.gz"
        label_path = pid_dir / f"{pid}_gt.nii.gz"
        if not image_path.exists():
            continue

        row = {"pid": pid, "image": _rel(image_path, base), "label": _rel(label_path, base)}
        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)
        rows.append(row)

    return pd.DataFrame(rows)


def build_mnms_ds_split(split_dir, metadata_csv=None, base_dir=None):
    """Build M&Ms split (SAX ED/ES required, SAX time-series optional)."""
    split_dir = Path(split_dir)
    base = Path(base_dir) if base_dir else split_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(split_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        sax_ed = pid_dir / f"{pid}_sax_ed.nii.gz"
        sax_es = pid_dir / f"{pid}_sax_es.nii.gz"
        if not (sax_ed.exists() and sax_es.exists()):
            continue

        row = {
            "pid": pid,
            "sax_ed": _rel(sax_ed, base),
            "sax_ed_gt": _rel(pid_dir / f"{pid}_sax_ed_gt.nii.gz", base),
            "sax_es": _rel(sax_es, base),
            "sax_es_gt": _rel(pid_dir / f"{pid}_sax_es_gt.nii.gz", base),
            "sax_t": _rel(pid_dir / f"{pid}_sax_t.nii.gz", base),
        }
        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)
        rows.append(row)

    return pd.DataFrame(rows)


def build_mnms2_ds_split(split_dir, metadata_csv=None, base_dir=None):
    """Build M&Ms2 split (SAX + optional LAX 4C/2C)."""
    split_dir = Path(split_dir)
    base = Path(base_dir) if base_dir else split_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(split_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        sax_ed = pid_dir / f"{pid}_sax_ed.nii.gz"
        sax_es = pid_dir / f"{pid}_sax_es.nii.gz"
        if not (sax_ed.exists() and sax_es.exists()):
            continue

        row = {"pid": pid}
        for suffix in [
            "sax_ed", "sax_ed_gt", "sax_es", "sax_es_gt",
            "lax_4c_ed", "lax_4c_ed_gt", "lax_4c_es", "lax_4c_es_gt",
            "lax_2c_ed", "lax_2c_ed_gt", "lax_2c_es", "lax_2c_es_gt",
        ]:
            row[suffix] = _rel(pid_dir / f"{pid}_{suffix}.nii.gz", base)

        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)
        rows.append(row)

    return pd.DataFrame(rows)


def build_myops2020_ds_split(split_dir, metadata_csv=None, base_dir=None):
    """Build MyoPS2020 split (c0/de/t2 required, label optional)."""
    split_dir = Path(split_dir)
    base = Path(base_dir) if base_dir else split_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(split_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        c0 = pid_dir / f"{pid}_c0.nii.gz"
        de = pid_dir / f"{pid}_de.nii.gz"
        t2 = pid_dir / f"{pid}_t2.nii.gz"
        if not (c0.exists() and de.exists() and t2.exists()):
            continue

        row = {
            "pid": pid,
            "c0": _rel(c0, base), "de": _rel(de, base), "t2": _rel(t2, base),
            "label": _rel(pid_dir / f"{pid}_gt.nii.gz", base),
        }
        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)
        rows.append(row)

    return pd.DataFrame(rows)


def build_kaggle_ds_split(split_dir, metadata_csv=None, base_dir=None):
    """Build Kaggle split (SAX required, LAX 2C/4C optional)."""
    split_dir = Path(split_dir)
    base = Path(base_dir) if base_dir else split_dir.parent

    meta_df = pd.read_csv(metadata_csv) if metadata_csv else None
    if meta_df is not None and "pid" in meta_df.columns:
        meta_df["pid"] = meta_df["pid"].astype(str)
        meta_dict = meta_df.set_index("pid").to_dict(orient="index")
        meta_columns = list(meta_df.columns.drop("pid"))
    else:
        meta_dict = {}
        meta_columns = []

    rows = []
    for pid_dir in sorted(split_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        sax_t = pid_dir / f"{pid}_sax_t.nii.gz"
        if not sax_t.exists():
            continue

        row = {
            "pid": pid,
            "sax_t": _rel(sax_t, base),
            "lax_2c_t": _rel(pid_dir / f"{pid}_lax_2c_t.nii.gz", base),
            "lax_4c_t": _rel(pid_dir / f"{pid}_lax_4c_t.nii.gz", base),
        }
        meta_row = meta_dict.get(pid, {})
        for col in meta_columns:
            row[col] = meta_row.get(col, None)
        rows.append(row)

    return pd.DataFrame(rows)

def push_folder_to_hub(repo_id: str, dataset_dir: Path, readme_path: Path | None = None) -> None:
    """Upload a processed dataset directory to Hugging Face Hub.

    Uses ``upload_large_folder`` which supports chunked uploads, automatic
    retries, and resumption — suitable for large NIfTI datasets.

    Args:
        repo_id: HuggingFace repo id (e.g. "user/cardiac_cine_acdc").
        dataset_dir: local directory containing split folders + metadata CSVs.
        readme_path: optional path to a README.md to upload.
    """
    api = HfApi()

    # Create the repo (no-op if it already exists)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    # Upload README first (small file, use regular upload)
    if readme_path and readme_path.exists():
        upload_readme(repo_id, readme_path)

    # Upload the dataset folder with large-folder strategy
    print(f"  Uploading NIfTI files from {dataset_dir} ...")
    api.upload_large_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["**/*.nii.gz", "*.csv", "**/*.csv"],
    )
    print("  ✓ NIfTI files + metadata CSVs uploaded")


# ---------------------------------------------------------------------------
# Build helpers:  dataset name → (build_function, splits_with_metadata)
# ---------------------------------------------------------------------------
_SPLIT_NAMES = ["train", "val", "test"]


def _build_dataset_csvs(dataset_name: str, dataset_dir: Path) -> list[Path]:
    """Build per-split CSVs with relative file paths for *dataset_name*.

    Returns a list of CSV paths that were written (one per split that exists).
    """
    builders = {
        "acdc":     build_acdc_split,
        "emidec":   build_emidec_ds,
        "mnms":     build_mnms_ds_split,
        "mnms2":    build_mnms2_ds_split,
        "myops2020": build_myops2020_ds_split,
        "kaggle":   build_kaggle_ds_split,
    }
    build_fn = builders.get(dataset_name)
    if build_fn is None:
        print(f"  ⚠ No builder for '{dataset_name}', skipping CSV generation")
        return []

    written: list[Path] = []
    for split in _SPLIT_NAMES:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue

        meta_csv = dataset_dir / f"{split}_metadata.csv"
        meta_csv = meta_csv if meta_csv.exists() else None

        print(f"  Building {split} manifest …")
        df = build_fn(split_dir, metadata_csv=meta_csv, base_dir=dataset_dir)

        out_csv = dataset_dir / f"{split}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  ✓ {split}.csv  ({len(df)} rows)")
        written.append(out_csv)

    return written


def main(username: str = None,
         data_dir: str = None,
         output_dir: str = "./CardiacCineMA",
         datasets: list = None, split: str = "train",
         shuffle_seed: int = 42, push_to_hub: bool = True, upload_readme_to_hub: bool = True):
    if push_to_hub and not username:
        raise ValueError("Username is required when push_to_hub=True")

    if datasets is None:
        datasets = ["acdc", "emidec", "kaggle", "mnms", "mnms2", "myops2020"]

    base = Path(data_dir)

    for dataset in datasets:
        dataset_dir = base / dataset
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*60}")

        # 1. Generate per-split CSVs with relative paths
        _build_dataset_csvs(dataset, dataset_dir)

        # 2. Optionally push to Hub
        if push_to_hub:
            repo_id = f"{username}/cardiac_cine_{dataset}"
            readme_path = dataset_dir / "README.md" if upload_readme_to_hub else None
            print(f"\n  🔹 Uploading {dataset} → {repo_id}")
            try:
                push_folder_to_hub(repo_id, dataset_dir, readme_path)
                print(f"  ✓ Upload complete! https://huggingface.co/datasets/{repo_id}")
            except Exception as e:
                print(f"  ✗ Error uploading {dataset}: {e}")
                raise

    if not push_to_hub:
        print("\n✓ Skipping Hub upload (CSVs saved locally)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and upload the CardiacCineMA Cine dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/vincent/MyWorkspace/UCF/CardioVLM/CardiacVLM-Cine-DS/Data/processed_output",
        help="Local preprocessed dataset directory"
    )

    parser.add_argument(
        "--username",
        type=str,
        default="viennh2012",
        help="Your Hugging Face username (required if --push-to-hub is set)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./CardiacCineMA",
        help="Local directory to save the dataset (default: ./CardiacCineMA)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["acdc", "emidec", "kaggle", "mnms", "mnms2", "myops2020"],
        default=None,
        help="Datasets to include (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip uploading to Hugging Face Hub (only save locally)"
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip Hugging Face login check (use if already authenticated)"
    )
    parser.add_argument(
        "--no-upload-readme",
        action="store_true",
        help="Skip uploading README.md to the Hub"
    )

    args = parser.parse_args()

    # Check authentication if pushing to hub
    push_to_hub = not args.no_push
    if push_to_hub:
        if not args.username:
            parser.error("--username is required when pushing to Hub (or use --no-push to skip upload)")

        if not args.skip_login:
            try:
                from huggingface_hub import login

                print("Checking Hugging Face authentication...")
                login()  # Opens CLI login prompt if not already authenticated
                print("✓ Authentication successful")
            except Exception as e:
                print(f"⚠ Warning: Could not verify authentication: {e}")
                print("Please run `huggingface-cli login` manually if upload fails.")

    try:
        main(
            username=args.username,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            datasets=args.datasets,
            split=args.split,
            shuffle_seed=args.shuffle_seed,
            push_to_hub=push_to_hub,
            upload_readme_to_hub=not args.no_upload_readme
        )
        print("\n✓ All done!")
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)