"""HuggingFace dataset download utility for the VQA pipeline.

Downloads preprocessed cardiac cine MRI datasets from HuggingFace Hub
to a local cache directory, preserving the directory structure that the
VQA ground-truth generation stage expects.

Includes automatic retry with exponential backoff for rate-limit (429)
errors, which are common when downloading without authentication.
"""

from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from cardio.utils.logging import get_logger

logger = get_logger(__name__)

HF_REPO_MAP: dict[str, str] = {
    "acdc": "viennh2012/cardiac_cine_acdc",
    "mnms": "viennh2012/cardiac_cine_mnms",
    "mnms2": "viennh2012/cardiac_cine_mnms2",
}

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 60


def _download_with_retry(
    repo_id: str,
    cache_dir: str | None,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """Download a single HF dataset repo with retry on rate-limit errors.

    ``snapshot_download`` is resumable: already-fetched files are skipped
    on subsequent calls, so retrying after a 429 picks up where it left off.

    Args:
        repo_id: HuggingFace repository identifier.
        cache_dir: override for the HF cache directory.
        max_retries: maximum number of retry attempts.

    Returns:
        Local path to the downloaded snapshot.

    Raises:
        HfHubHTTPError: if all retries are exhausted.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            return Path(local_path)
        except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
            last_err = exc
            # The 429 may be in the exception itself or in the chained cause
            err_chain = str(exc) + str(exc.__cause__ or "")
            if "429" in err_chain:
                logger.warning(
                    f"Rate-limited downloading {repo_id} "
                    f"(attempt {attempt}/{max_retries}). "
                    f"Waiting {backoff}s before retry ..."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
            else:
                raise

    raise last_err  # type: ignore[misc]


def download_datasets(
    datasets: list[str] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Download HuggingFace datasets and return local paths.

    Each dataset is downloaded via ``snapshot_download`` which caches
    the result.  Subsequent calls with the same *cache_dir* are nearly
    instantaneous because HF Hub checks the local snapshot first.

    Args:
        datasets: dataset names to download (default: all three).
        cache_dir: root directory for the HF cache.  When ``None`` the
            default ``~/.cache/huggingface/hub`` is used.

    Returns:
        Dict mapping dataset name to the local snapshot path, e.g.
        ``{"acdc": Path("/Users/.../.cache/huggingface/hub/datasets--viennh2012--cardiac_cine_acdc/snapshots/abc123")}``.
    """
    if datasets is None:
        datasets = list(HF_REPO_MAP.keys())

    cache_str = str(cache_dir) if cache_dir else None

    result: dict[str, Path] = {}
    for name in datasets:
        repo_id = HF_REPO_MAP.get(name)
        if repo_id is None:
            logger.warning(f"Unknown dataset '{name}', skipping. Known: {list(HF_REPO_MAP)}")
            continue

        logger.info(f"Downloading {repo_id} (cached if already present) ...")
        try:
            local_path = _download_with_retry(repo_id, cache_str)
            logger.info(f"  {name} -> {local_path}")
            result[name] = local_path
        except (HfHubHTTPError, LocalEntryNotFoundError):
            logger.error(
                f"Failed to download {repo_id} after {MAX_RETRIES} retries. "
                f"Skipping {name}. Consider logging in with `huggingface-cli login` "
                f"to avoid rate limits."
            )

    if not result:
        msg = "No datasets could be downloaded. Check your internet connection or HF authentication."
        raise RuntimeError(msg)

    return result
