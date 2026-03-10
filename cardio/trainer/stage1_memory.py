"""Stage 1 memory-formation supervised loss.

Computes the memory-augmented loss ``L_mem`` and a frozen baseline loss
``L_base`` (without memory).  The overall Stage 1 objective is
``L_mem - sg(L_base)`` where ``sg`` is stop-gradient.

Ported from ``_reference/CineMem/main/trainer/stage1_memory_formation.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_VISUAL_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
)


def stage1_loss(
    base_model: Any,  # noqa: ANN401
    cinemem_model: Any,  # noqa: ANN401
    inputs: dict[str, Any],
    target_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute memory-formation loss and frozen baseline loss.

    Args:
        base_model: the frozen base VLM (e.g. Qwen2.5-VL).
        cinemem_model: the ``CineMemModel`` wrapper whose memory
            parameters are being trained.
        inputs: processor outputs (``input_ids``, ``attention_mask``,
            and visual tensors).
        target_text: ground-truth answer text.

    Returns:
        ``(loss_mem, loss_base)`` — both scalar tensors.
    """
    tokenizer = cinemem_model.tokenizer
    device = cinemem_model.device

    tgt_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    visual_kw = {k: inputs[k] for k in _VISUAL_KEYS if k in inputs}

    # --- Baseline loss (frozen, no memory) --------------------------------
    with torch.no_grad():
        full_ids = torch.cat([inputs["input_ids"], tgt_ids], dim=1)
        attn = torch.ones_like(full_ids, dtype=torch.long)
        labels = full_ids.clone()
        labels[:, : inputs["input_ids"].size(1)] = -100

        out = base_model(input_ids=full_ids, attention_mask=attn, **visual_kw)
        logits = out.logits[:, :-1, :]
        loss_base = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    # --- Memory-augmented loss --------------------------------------------
    base_out = base_model(**inputs, output_hidden_states=True)
    hidden = base_out.hidden_states[-1]  # (B, T, D)
    mem = cinemem_model.form_memory(hidden, mem_type="short")

    prompt_embeds = base_out.hidden_states[0][:, : inputs["input_ids"].size(1), :]
    tgt_embeds = base_model.get_input_embeddings()(tgt_ids)
    inp_embeds = torch.cat([prompt_embeds, mem, tgt_embeds], dim=1)
    attn2 = torch.ones(inp_embeds.size()[:-1], device=device, dtype=torch.long)

    mem_len = mem.size(1)
    labels2 = torch.cat(
        [
            inputs["input_ids"],
            torch.full(
                (inputs["input_ids"].size(0), mem_len),
                -100,
                device=device,
                dtype=torch.long,
            ),
            tgt_ids,
        ],
        dim=1,
    )
    labels2[:, : inputs["input_ids"].size(1) + mem_len] = -100

    out2 = base_model(inputs_embeds=inp_embeds, attention_mask=attn2)
    logits2 = out2.logits[:, :-1, :]
    loss_mem = F.cross_entropy(
        logits2.reshape(-1, logits2.size(-1)),
        labels2[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    return loss_mem, loss_base
