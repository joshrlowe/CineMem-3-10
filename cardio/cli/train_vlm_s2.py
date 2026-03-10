r"""Stage 2 NGRPO training with composite cardiac rewards.

Performs Negative-enhanced Group Relative Policy Optimisation (NGRPO)
with the full :class:`~cardio.trainer.rewards.composite.CompositeRewardEngine`
pipeline, token-level penalty masks (TR-GRPO), and asymmetric clipping.

Fixes vs. reference (``_reference/CineMem/main/cli/train_stage2_cardiac.py``):

1. **Indentation bug** -- loss computation, optimizer step, checkpoint
   save, and logging are all correctly inside the inner training loop.
2. **``global reward_ema``** -- eliminated; reward EMA is tracked as an
   instance variable inside :class:`~cardio.trainer.grpo.NGRPOTrainer`.
3. **Gradient accumulation** -- properly implemented with ``GradScaler``.
4. **Mixed precision** -- ``torch.autocast`` + ``GradScaler``.
5. **G rollouts** -- generates *G* rollouts per prompt for proper group
   advantage estimation (reference only generated 1 + 1 reverse).

Usage::

    cardio-train-vlm-s2 \
        --train_jsonl data/vqa_jsonl/train.jsonl \
        --image-dir /path/to/images \
        --init_from outputs/stage1/epoch0 \
        --output_dir outputs/stage2 \
        --epochs 1
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import optim
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

from cardio.data.collate import build_processor_inputs, collate_samples
from cardio.data.datasets.jsonl_vl import JsonlVLDataset
from cardio.trainer.grpo import NGRPOTrainer
from cardio.trainer.optim import GradScaler
from cardio.trainer.rewards.cardiac import cardiac_reward_normalised
from cardio.trainer.rewards.composite import CompositeRewardEngine
from cardio.trainer.rewards.dcr import AutoMetricConverter, DivideConquerEvaluator
from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier
from cardio.trainer.rewards.vprm import ACCAHAVerifier
from cardio.utils import ensure_dir, get_logger, init_wandb, set_seed, to_torch_dtype
from cardio.utils.misc import load_yaml
from cardio.vlm import CineMemModel, load_qwen25vl
from cardio.vlm.config import build_cinemem_config

logger = get_logger(__name__)


# -----------------------------------------------------------------------
# Gradient masking for memory-token embeddings
# -----------------------------------------------------------------------


def _build_grad_hook(
    cinemem: CineMemModel,
    end_lr_mult: float = 0.1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a gradient hook that masks all but memory-token embedding rows.

    End-token gradients are additionally scaled down by *end_lr_mult*.
    """
    special_ids: list[int] = [
        cinemem.short_invoke_id,
        cinemem.short_end_id,
        cinemem.long_invoke_id,
        cinemem.long_end_id,
    ]
    end_ids: list[int] = [cinemem.short_end_id, cinemem.long_end_id]

    if cinemem.tdm_invoke_id is not None:
        special_ids.append(cinemem.tdm_invoke_id)
        special_ids.append(cinemem.tdm_end_id)  # type: ignore[arg-type]
        end_ids.append(cinemem.tdm_end_id)  # type: ignore[arg-type]
    if cinemem.psm_invoke_id is not None:
        special_ids.append(cinemem.psm_invoke_id)
        special_ids.append(cinemem.psm_end_id)  # type: ignore[arg-type]
        end_ids.append(cinemem.psm_end_id)  # type: ignore[arg-type]

    special_t = torch.tensor(special_ids, device=cinemem.device, dtype=torch.long)
    end_t = torch.tensor(end_ids, device=cinemem.device, dtype=torch.long)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(grad)
        g[special_t] = grad[special_t]
        g[end_t] *= end_lr_mult
        return g

    return hook


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    """Entry point for ``cardio-train-vlm-s2``."""
    ap = argparse.ArgumentParser(description="Stage 2 NGRPO training")
    ap.add_argument("--config", default="configs/cinemem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument(
        "--init_from", default=None,
        help="Stage 1 checkpoint folder containing main.pt",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=0.02)
    ap.add_argument("-G", "--group_size", type=int, default=4)
    ap.add_argument("--r_max", type=float, default=1.0)
    ap.add_argument("--epsilon_neg", type=float, default=0.16)
    ap.add_argument("--epsilon_pos", type=float, default=0.24)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
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

    if args.init_from is not None:
        state = torch.load(
            Path(args.init_from) / "main.pt",
            map_location="cpu",
        )
        cinemem.load_state_dict(state["cinemem_state"], strict=False)

    # Freeze everything, then selectively unfreeze embedding rows via hook
    for p in cinemem.parameters():
        p.requires_grad = False

    emb = cinemem.base_model.get_input_embeddings()
    emb.weight.requires_grad = True
    emb.weight.register_hook(_build_grad_hook(cinemem))
    opt = optim.AdamW([emb.weight], lr=args.lr)

    # --- Reference model (for KL) ----------------------------------------
    ref_model = None
    try:
        ref_model = copy.deepcopy(cinemem.base_model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    except Exception:  # noqa: BLE001
        logger.warning("Could not deepcopy base_model for KL reference; proceeding without KL.")

    # --- Trainer ----------------------------------------------------------
    trainer = NGRPOTrainer(
        cinemem,
        ref_model=ref_model,
        kl_beta=args.kl_beta,
        r_max=args.r_max,
        epsilon_neg=args.epsilon_neg,
        epsilon_pos=args.epsilon_pos,
    )

    # --- Reward engine ----------------------------------------------------
    composite_engine = CompositeRewardEngine(
        cardiac_verifier=cardiac_reward_normalised,
        acc_aha_verifier=ACCAHAVerifier(),
        memory_verifier=MemoryInvocationVerifier(),
        dcr_evaluator=DivideConquerEvaluator(),
        amc=AutoMetricConverter(),
    )

    # --- Data / AMP -------------------------------------------------------
    ds = JsonlVLDataset(args.train_jsonl, args.image_dir or "")
    ensure_dir(args.output_dir)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = GradScaler()
    max_tok = int(cfg_dict.get("training", {}).get("max_new_tokens", 256))
    grad_accum_steps = args.grad_accum_steps
    max_grad_norm = args.max_grad_norm
    group_size = args.group_size

    # --- W&B --------------------------------------------------------------
    wandb_run, _ = init_wandb(
        project=args.wandb_project,
        config={
            **cfg_dict,
            "lr": args.lr,
            "kl_beta": args.kl_beta,
            "group_size": group_size,
            "r_max": args.r_max,
            "epsilon_neg": args.epsilon_neg,
            "epsilon_pos": args.epsilon_pos,
            "grad_accum_steps": grad_accum_steps,
        },
        run_name=args.run_name,
    )

    # --- Training ---------------------------------------------------------
    cinemem.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Stage2 epoch {epoch}")
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
            prompt_ids = inputs["input_ids"]

            # --- Generate G rollouts ------------------------------------
            rewards_list: list[float] = []
            token_masks: list[torch.Tensor | None] = []
            gen_texts: list[str] = []
            gen_id_list: list[torch.Tensor] = []

            for _g in range(group_size):
                pred_out = cinemem.generate(
                    images=[img],
                    prompts=[prompt],
                    max_new_tokens=max_tok,
                    temperature=args.temperature,
                    enable_cinemem=True,
                    return_token_ids=True,
                    skip_special_tokens=False,
                )
                if isinstance(pred_out, tuple):
                    pred_text, gen_ids = pred_out
                else:
                    pred_text = pred_out
                    gen_ids = None

                pred = pred_text[0]
                pred_clean = (
                    tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    if gen_ids is not None
                    else re.sub(r"<[^>]+>", "", pred).strip()
                )

                invocation_log = list(cinemem.invocation_log)
                category = batch.get("categories", ["general"])[0] if isinstance(batch, dict) else "general"

                result = composite_engine.compute(
                    generated_text=pred,
                    predicted_answer=pred_clean,
                    ground_truth=answer,
                    category=category,
                    invocation_log=invocation_log,
                    predicted_bboxes=None,
                    ground_truth_masks=None,
                    token_offsets=None,
                    seq_len=None,
                )

                rewards_list.append(result["composite_reward"])
                token_masks.append(result["token_penalty_mask"])
                gen_texts.append(pred_clean)
                if gen_ids is not None:
                    gen_id_list.append(gen_ids[0] if gen_ids.dim() > 1 else gen_ids)

            rewards_tensor = torch.tensor(
                rewards_list, device=cinemem.device, dtype=torch.float32,
            )
            trainer.update_reward_ema(rewards_tensor.mean().item())

            # Use best rollout for policy gradient
            best_idx = int(rewards_tensor.argmax().item())
            best_text = gen_texts[best_idx]
            best_mask = token_masks[best_idx]

            sampled_ids = tokenizer(
                best_text, return_tensors="pt",
            ).input_ids.to(cinemem.device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                loss = trainer.loss_from_samples(
                    {"input_ids": prompt_ids},
                    sampled_ids,
                    rewards_tensor,
                    token_penalty_mask=best_mask,
                )

            should_step = (i + 1) % grad_accum_steps == 0
            grad_norm = scaler(
                loss / grad_accum_steps,
                optimizer=opt,
                parameters=[emb.weight],
                clip_grad=max_grad_norm,
                update_grad=should_step,
            )

            if should_step:
                opt.zero_grad()
                global_step += 1

            mean_r = float(rewards_tensor.mean().cpu())
            pbar.set_postfix({
                "reward": f"{mean_r:.3f}",
                "loss": f"{float(loss.detach().cpu()):.4f}",
            })

            if wandb_run is not None:
                import wandb

                log_dict: dict = {
                    "train/reward_mean": mean_r,
                    "train/reward_std": float(rewards_tensor.std().cpu()),
                    "train/reward_best": float(rewards_tensor.max().cpu()),
                    "train/loss": float(loss.detach().cpu()),
                    "train/reward_ema": trainer.reward_ema,
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

    logger.info("Stage 2 NGRPO training complete.")


if __name__ == "__main__":
    main()
