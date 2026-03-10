r"""Stage 1 training: memory-formation supervised fine-tuning.

Minimises ``L_mem - sg(L_base)`` so that the memory modules learn to
produce embeddings that *reduce* next-token prediction loss relative to
the frozen base model.

Fixes vs. reference (``_reference/CineMem/main/cli/train_stage1.py``):

1. **Gradient accumulation** -- the reference config declares
   ``grad_accum: 4`` but never implements it.  This script accumulates
   gradients properly and only steps every ``--grad_accum_steps`` micro-
   steps.
2. **Mixed precision** -- wraps forward passes in ``torch.autocast`` and
   uses :class:`~cardio.trainer.optim.GradScaler` for safe FP16/BF16
   backward + optimizer step.
3. **W&B logging** -- optional via ``--wandb_project``.

Usage::

    cardio-train-vlm-s1 \
        --train_jsonl data/vqa_jsonl/train.jsonl \
        --image-dir /path/to/images \
        --output_dir outputs/stage1 \
        --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from cardio.data.collate import build_processor_inputs, collate_samples
from cardio.data.datasets.jsonl_vl import JsonlVLDataset
from cardio.trainer.optim import GradScaler
from cardio.trainer.stage1_memory import stage1_loss
from cardio.utils import ensure_dir, get_logger, init_wandb, set_seed, to_torch_dtype
from cardio.utils.misc import load_yaml
from cardio.vlm import CineMemModel, load_qwen25vl
from cardio.vlm.config import build_cinemem_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``cardio-train-vlm-s1``."""
    ap = argparse.ArgumentParser(description="Stage 1 memory-formation training")
    ap.add_argument("--config", default="configs/cinemem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args(argv)

    # --- Config -----------------------------------------------------------
    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_cinemem_config(cfg_dict)

    seed = int(cfg_dict.get("training", {}).get("seed", 42))
    set_seed(seed)

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    # --- Model ------------------------------------------------------------
    base_model, tokenizer, processor = load_qwen25vl(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust,
    )
    cinemem = CineMemModel(base_model, tokenizer, processor, viscfg)

    for p in cinemem.base_model.parameters():
        p.requires_grad = False

    trainable = [p for p in cinemem.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else float(
        cfg_dict.get("training", {}).get("lr", 2e-4)
    )
    opt = optim.AdamW(trainable, lr=lr)

    # --- Data -------------------------------------------------------------
    ds = JsonlVLDataset(args.train_jsonl, args.image_dir or "")
    ensure_dir(args.output_dir)

    # --- AMP / scaler -----------------------------------------------------
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = GradScaler()

    grad_accum_steps = args.grad_accum_steps
    max_grad_norm = args.max_grad_norm

    # --- W&B --------------------------------------------------------------
    wandb_run, _ = init_wandb(
        project=args.wandb_project,
        config={**cfg_dict, "lr": lr, "grad_accum_steps": grad_accum_steps},
        run_name=args.run_name,
    )

    # --- Training ---------------------------------------------------------
    cinemem.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Stage1 epoch {epoch}")
        opt.zero_grad()

        for i in pbar:
            batch = collate_samples([ds[i]])
            img = batch["images"][0]
            prompt = batch["prompts"][0]
            answer = batch["answers"][0]
            if answer is None:
                continue

            inputs = build_processor_inputs(processor, img, prompt)
            inputs = {
                k: v.to(cinemem.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                loss_mem, loss_base = stage1_loss(
                    cinemem.base_model, cinemem, inputs, answer,
                )
                loss = loss_mem - loss_base.detach()

            should_step = (i + 1) % grad_accum_steps == 0
            grad_norm = scaler(
                loss / grad_accum_steps,
                optimizer=opt,
                parameters=trainable,
                clip_grad=max_grad_norm,
                update_grad=should_step,
            )

            if should_step:
                opt.zero_grad()
                global_step += 1

            pbar.set_postfix({
                "loss_mem": f"{float(loss_mem.detach().cpu()):.4f}",
                "loss_base": f"{float(loss_base.detach().cpu()):.4f}",
            })

            if wandb_run is not None:
                import wandb

                log_dict: dict = {
                    "train/loss_mem": float(loss_mem.detach().cpu()),
                    "train/loss_base": float(loss_base.detach().cpu()),
                    "train/loss": float(loss.detach().cpu()),
                    "train/lr": opt.param_groups[0]["lr"],
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = float(grad_norm)
                wandb.log(log_dict, step=global_step)

        # --- Checkpoint ---------------------------------------------------
        ckpt = Path(args.output_dir) / f"epoch{epoch}"
        ensure_dir(str(ckpt))
        torch.save(
            {"cinemem_state": cinemem.state_dict(), "config": cfg_dict},
            ckpt / "main.pt",
        )
        tokenizer.save_pretrained(str(ckpt))
        logger.info(f"Saved checkpoint to {ckpt}")

    logger.info("Stage 1 training complete.")


if __name__ == "__main__":
    main()
