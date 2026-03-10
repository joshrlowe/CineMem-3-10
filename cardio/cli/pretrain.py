"""MAE pre-training on UK Biobank cardiac cine MRI.

Ported from ``_reference/CineMA/cinema/mae/pretrain.py`` with DDP
support preserved.

Usage (Hydra)::

    cardio-pretrain --config-path /path/to --config-name config

Usage (multi-GPU)::

    cardio-pretrain ddp=true
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.data import Dataset
from monai.transforms import (
    Compose,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
)
from timm.optim import param_groups_weight_decay
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from cardio.data.constants import UKB_N_FRAMES
from cardio.trainer.optim import GradScaler, adjust_learning_rate, get_n_accum_steps, save_checkpoint
from cardio.utils.device import ddp_setup, get_amp_dtype_and_device, get_free_port, print_model_info, setup_ddp_model
from cardio.utils.logging import get_logger, init_wandb
from cardio.vision.mae import CineMA, get_model

if TYPE_CHECKING:
    from monai.transforms import Transform
    from omegaconf import DictConfig

logger = get_logger(__name__)


# =========================================================================
# Data loading
# =========================================================================


def scan_manifests(data_dirs: str | Path | list[str | Path], rescan: bool) -> list[Path]:
    """Scan for UK Biobank manifest files.

    Args:
        data_dirs: one or more dataset root directories.
        rescan: force a re-scan even if cached manifests exist.

    Returns:
        Sorted list of manifest file paths.
    """
    if isinstance(data_dirs, Path | str):
        data_dirs = [data_dirs]
    data_dirs = [Path(x) for x in data_dirs]
    manifest_paths: list[Path] = []

    if not rescan:
        for data_dir in data_dirs:
            json_path = Path(data_dir) / "manifest_paths.json"
            if not json_path.exists():
                rescan = True
                break
            with json_path.open(encoding="utf-8") as f:
                paths_to_add = [Path(x) for x in json.load(f)]
            if not paths_to_add[0].exists():
                rescan = True
                logger.warning("Manifest paths in %s are stale, re-scanning.", json_path)
                break
            manifest_paths += paths_to_add

    if rescan:
        manifest_paths = []
        for data_dir in data_dirs:
            logger.info("Scanning %s for manifest files.", data_dir)
            found = list(Path(data_dir).glob("**/*_manifest_sax.csv"))
            manifest_paths += found
            json_path = Path(data_dir) / "manifest_paths.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump([str(x) for x in found], f)
            logger.info("Found %d manifests in %s.", len(found), data_dir)

    logger.info("Total manifest files: %d.", len(manifest_paths))
    return sorted(manifest_paths)


def ukb_load_sample(manifest_path: Path, t: int) -> dict[str, np.ndarray]:
    """Load one UK Biobank sample at time-frame *t*.

    Args:
        manifest_path: path to the manifest CSV.
        t: time-frame index.

    Returns:
        Dict with keys ``sax``, ``lax_2c``, ``lax_3c``, ``lax_4c``.
    """
    eid_dir = manifest_path.parent
    eid = eid_dir.name
    data: dict[str, np.ndarray] = {}
    reader = sitk.ImageFileReader()

    for view in ["lax_2c", "lax_3c", "lax_4c", "sax"]:
        reader.SetFileName(str(eid_dir / f"{eid}_{view}.nii.gz"))
        reader.ReadImageInformation()
        size = list(reader.GetSize())
        if t >= size[-1]:
            t = size[-1] // 2
        size[-1] = 1
        reader.SetExtractIndex([0, 0, 0, t])
        reader.SetExtractSize(size)
        image = np.transpose(sitk.GetArrayFromImage(reader.Execute()))[..., 0]
        if view != "sax":
            image = image[..., 0]
        data[view] = image

    return data


class UKBDataset(Dataset):
    """UK Biobank dataset (50 frames per subject)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise with a per-instance RNG."""
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng()

    def _transform(self, index: int) -> dict[str, torch.Tensor]:
        """Load and transform a single sample at a random time-frame."""
        t = int(self.rng.integers(UKB_N_FRAMES))
        np_data = ukb_load_sample(self.data[index], t)
        data = {
            "sax": torch.from_numpy(np_data["sax"][None, ...]),
            "lax_2c": torch.from_numpy(np_data["lax_2c"][None, ...]),
            "lax_3c": torch.from_numpy(np_data["lax_3c"][None, ...]),
            "lax_4c": torch.from_numpy(np_data["lax_4c"][None, ...]),
        }
        return self.transform(data)


def get_transform(config: DictConfig) -> Transform:
    """Build the MONAI transform pipeline for pre-training.

    Args:
        config: OmegaConf configuration.

    Returns:
        A composed MONAI transform.
    """
    return Compose([
        RandZoomd(
            keys="sax", prob=config.transform.prob,
            mode="trilinear", padding_mode="constant", lazy=True, allow_missing_keys=True,
        ),
        RandZoomd(
            keys=("lax_2c", "lax_3c", "lax_4c"), prob=config.transform.prob,
            mode="bicubic", padding_mode="constant", lazy=True, allow_missing_keys=True,
        ),
        ScaleIntensityd(keys=("sax", "lax_2c", "lax_3c", "lax_4c")),
        SpatialPadd(
            keys="sax", spatial_size=config.data.sax.patch_size,
            method="end", lazy=True, allow_missing_keys=True,
        ),
        SpatialPadd(
            keys=("lax_2c", "lax_3c", "lax_4c"), spatial_size=config.data.lax.patch_size,
            method="end", lazy=True, allow_missing_keys=True,
        ),
    ])


# =========================================================================
# Training loop
# =========================================================================


def pretrain_one_epoch(
    model: CineMA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    n_accum_steps: int,
    world_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: DictConfig,
    epoch: int,
    n_samples: int,
    wandb_run: object | None,
) -> int:
    """Pre-train for one epoch.

    Returns:
        Updated sample counter.
    """
    views = config.model.views
    batch_size_per_step = config.train.batch_size_per_device * world_size
    enc_mask_ratio = config.train.enc_mask_ratio
    clip_grad = config.train.clip_grad if config.train.clip_grad > 0 else None

    for i, batch in enumerate(dataloader):
        lr = adjust_learning_rate(
            optimizer=optimizer,
            step=i / len(dataloader) + epoch,
            warmup_steps=config.train.n_warmup_epochs,
            max_n_steps=config.train.n_epochs,
            lr=config.train.lr,
            min_lr=config.train.min_lr,
        )
        with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            loss, _, _, metrics = model({k: v.to(device) for k, v in batch.items()}, enc_mask_ratio)
        metrics = {k: v.item() for k, v in metrics.items()}

        if torch.isnan(loss).any():
            logger.error("NaN loss, metrics: %s.", metrics)
            continue

        loss /= n_accum_steps
        update_grad = (i + 1) % n_accum_steps == 0
        grad_norm = loss_scaler(
            loss=loss, optimizer=optimizer, clip_grad=clip_grad,
            parameters=model.parameters(), update_grad=update_grad,
        )
        if update_grad:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_samples += batch_size_per_step
        if update_grad and wandb_run is not None:
            prefix = f"{views[0]}_" if len(views) == 1 else ""
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            metrics.update({"grad_norm": grad_norm.item(), "lr": lr, "n_samples": n_samples})
            wandb_run.log(metrics)

    return n_samples


def pretrain(  # noqa: C901
    rank: int,
    world_size: int,
    port: int,
    config: DictConfig,
) -> None:
    """Run MAE pre-training (single-process or DDP worker).

    Args:
        rank: unique process identifier.
        world_size: total number of processes.
        port: TCP port for distributed training.
        config: OmegaConf configuration.
    """
    if world_size > 1:
        ddp_setup(rank, world_size, port)
    amp_dtype, device = get_amp_dtype_and_device()

    seed = config.seed + rank
    torch.manual_seed(seed)

    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=world_size,
    )

    # ----- dataset --------------------------------------------------------
    manifest_paths = scan_manifests(config.data.dir, rescan=False)
    transform = get_transform(config)
    dataset = UKBDataset(data=manifest_paths, transform=transform)
    if config.data.max_n_samples > 0:
        n = min(config.data.max_n_samples, len(manifest_paths))
        dataset = Subset(dataset, np.arange(n))
        logger.info("Using %d samples for training.", n)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else RandomSampler(dataset)
    )
    dataloader = DataLoader(
        dataset=dataset, sampler=sampler,
        batch_size=config.train.batch_size_per_device,
        drop_last=True, pin_memory=True, num_workers=config.train.n_workers_per_device,
    )

    # ----- model ----------------------------------------------------------
    model = get_model(config)
    print_model_info(model)
    model, model_wo_ddp = setup_ddp_model(model=model, device=device, rank=rank, world_size=world_size)

    if world_size > 1:
        tmp_ckp_path = tempfile.gettempdir() + "/ckpt.pt"
        if rank == 0:
            torch.save(model.state_dict(), tmp_ckp_path)
            logger.info("Saved rank-0 weights to %s.", tmp_ckp_path)
        dist.barrier()
        if rank > 0:
            model.load_state_dict(
                torch.load(tmp_ckp_path, map_location={"cuda:0": f"cuda:{rank}"}, weights_only=True),
            )
            logger.info("Loaded rank-0 weights on rank %d.", rank)

    # ----- optimizer ------------------------------------------------------
    logger.info("Initializing optimizer.")
    param_groups = param_groups_weight_decay(model=model_wo_ddp, weight_decay=config.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    n_samples = 0
    start_epoch = 0
    if config.train.ckpt_path is not None:
        from cardio.trainer.optim import load_checkpoint

        model_wo_ddp, optimizer, loss_scaler, epoch, n_samples = load_checkpoint(
            ckpt_path=Path(config.train.ckpt_path),
            model_wo_ddp=model_wo_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )
        start_epoch = epoch + 1

    # ----- W&B ------------------------------------------------------------
    wandb_run, ckpt_dir = None, None
    if rank == 0:
        tags = ["ukb_mae_pretrain"]
        if len(config.model.views) > 1:
            tags.append("multi_view")
        wandb_run, ckpt_dir = init_wandb(config=config, tags=tags)

    # ----- training -------------------------------------------------------
    logger.info("Start training.")
    saved_ckpt_paths: list[Path] = []
    max_n_ckpts = config.train.max_n_ckpts
    model.train(True)

    for epoch in range(start_epoch, config.train.n_epochs):
        optimizer.zero_grad()
        n_samples = pretrain_one_epoch(
            model=model, dataloader=dataloader, optimizer=optimizer,
            loss_scaler=loss_scaler, n_accum_steps=n_accum_steps,
            world_size=world_size, device=device, amp_dtype=amp_dtype,
            config=config, epoch=epoch, n_samples=n_samples, wandb_run=wandb_run,
        )

        if rank != 0 or ckpt_dir is None:
            continue
        ckpt_path = save_checkpoint(ckpt_dir, epoch, model_wo_ddp, optimizer, loss_scaler, n_samples)
        saved_ckpt_paths.append(ckpt_path)
        logger.info("Saved checkpoint epoch %d at %s (%d samples).", epoch, ckpt_path, n_samples)

        if len(saved_ckpt_paths) <= max_n_ckpts or max_n_ckpts <= 0:
            continue
        to_delete = saved_ckpt_paths.pop(0)
        ckpt_epoch = int(to_delete.stem.split("_")[1])
        if (ckpt_epoch + 1) % 100 == 0:
            continue
        to_delete.unlink(missing_ok=True)
        logger.info("Deleted outdated checkpoint %s.", to_delete)

    if world_size > 1:
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entry point for MAE pre-training."""
    world_size = torch.cuda.device_count()
    port = get_free_port()
    if world_size > 1 and config.ddp:
        mp.spawn(pretrain, args=(world_size, port, config), nprocs=world_size)
    else:
        if world_size > 1:
            logger.warning("world_size=%d > 1 but DDP is disabled.", world_size)
        pretrain(rank=0, world_size=1, port=port, config=config)


if __name__ == "__main__":
    main()
