"""Load Qwen2.5-VL with CardioVLM special tokens.

Handles tokenizer extension, embedding initialisation, and device
dispatch quirks (MPS eager loading, bfloat16 fallback).
"""

from __future__ import annotations

import torch

from cardio.vlm.constants import ALL_SPECIAL_TOKENS


def add_tokens(tokenizer):
    """Register all CardioVLM special tokens on *tokenizer*."""
    tokenizer.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    return tokenizer


def init_token_embeddings(
    model,
    tokenizer,
    init_from_token: str | None = None,
    noise_std: float = 1e-3,
) -> None:
    """Initialise embeddings for newly-added special tokens.

    Each new token is set to the embedding of a reference token (defaulting
    to ``eos_token``) plus a small Gaussian perturbation so that they start
    near a valid region of the embedding space but are distinguishable.
    Output embeddings (lm_head) are synchronised when they are not tied.
    """
    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        return
    w_in = emb_layer.weight

    init_id = None
    if init_from_token is not None:
        init_id = tokenizer.convert_tokens_to_ids(init_from_token)
    if init_id is None or init_id == tokenizer.unk_token_id:
        init_id = tokenizer.eos_token_id
    if init_id is None:
        return

    with torch.no_grad():
        ref = w_in[init_id].detach().clone()
        for tok in ALL_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                continue
            w_in[tid].copy_(ref + torch.randn_like(ref) * noise_std)

        out_layer = getattr(model, "get_output_embeddings", lambda: None)()
        if out_layer is not None and out_layer.weight.data_ptr() != w_in.data_ptr():
            w_out = out_layer.weight
            for tok in ALL_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid == tokenizer.unk_token_id:
                    continue
                w_out[tid].copy_(w_in[tid])


def _resolve_auto_model_class():
    """Pick the right Auto class for Qwen2.5-VL (varies across transformers versions)."""
    import transformers

    for cls_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            return cls
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def load_qwen25vl(
    model_name_or_path: str,
    torch_dtype=None,
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
):
    """Load a Qwen2.5-VL model, tokenizer, and processor.

    Special tokens are injected automatically and their embeddings are
    initialised from the ``eos_token`` embedding with small noise.

    On Apple Silicon (MPS) the model is loaded eagerly on CPU and moved
    to MPS afterwards, working around ``accelerate`` meta-tensor issues.
    ``bfloat16`` is downcast to ``float16`` on MPS since MPS does not
    support bfloat.

    Returns:
        ``(model, tokenizer, processor)`` tuple.
    """
    from transformers import AutoProcessor, AutoTokenizer

    auto_model_cls = _resolve_auto_model_class()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    old_vocab = len(tokenizer)
    tokenizer = add_tokens(tokenizer)

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    move_to: str | None = None

    if not has_cuda and device_map is not None and device_map != "cpu":
        device_map = None
        if has_mps:
            move_to = "mps"
            if torch_dtype == torch.bfloat16:
                torch_dtype = torch.float16

    load_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "low_cpu_mem_usage": False,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    model = auto_model_cls.from_pretrained(model_name_or_path, **load_kwargs)

    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if len(tokenizer) > old_vocab:
        init_token_embeddings(model, tokenizer, init_from_token=None, noise_std=1e-3)

    if move_to is not None:
        model = model.to(move_to)

    return model, tokenizer, processor
