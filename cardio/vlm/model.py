"""CineMemModel — memory-augmented vision-language model.

Ported from ``_reference/CineMem/main/model/model.py`` with the following
changes:

**Bug fixes**

- :func:`select_visual_positions` (extracted as a standalone, independently
  testable function) uses :func:`torch.where` to find *all*
  ``<|vision_start|>`` / ``<|vision_end|>`` pairs, replacing the fragile
  ``list.index`` fallback that silently dropped multi-segment inputs.

**Extensions**

- Dual-memory support: when ``config.use_dual_memory`` is ``True``, the
  model instantiates a :class:`~cardio.vlm.dual_memory.DualMemoryManager`
  (TDM + PSM) instead of the legacy short/long formers.
- :meth:`CineMemModel._build_H` accepts optional ``cinema_features``
  (from CineMA ConvViT/ConvUNetR encoders), projects them to match the
  Qwen hidden dimension, and concatenates them into the hidden-state
  matrix used for memory formation.
- :meth:`CineMemModel.form_memory` routes to the correct memory bank
  (dual or legacy) based on the ``mem_type`` label.
- :meth:`CineMemModel.generate` tracks every memory invocation in
  ``self.invocation_log`` for downstream reward-model consumption.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from cardio.data.collate import build_processor_inputs
from cardio.utils.logging import get_logger
from cardio.vlm.config import CineMemConfig
from cardio.vlm.dual_memory import DualMemoryManager
from cardio.vlm.lora_utils import (
    is_peft_available,
    make_lora_adapters,
    set_active_adapter,
)
from cardio.vlm.memory_former import TinyMemoryFormer
from cardio.vlm.mrope_fix import TemporalAlignmentOverride
from cardio.vlm.query_builder import QueryBuilder

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Standalone visual-position detection (torch.where fix)
# -----------------------------------------------------------------------

_MEM_TYPE_DUAL_MAP = {"short": "tdm", "long": "psm", "tdm": "tdm", "psm": "psm"}
_MEM_TYPE_LEGACY_MAP = {"short": "short", "long": "long", "tdm": "short", "psm": "long"}


def select_visual_positions(
    input_ids: torch.LongTensor,
    vision_start_id: int,
    vision_end_id: int,
) -> list[list[tuple[int, int]]]:
    """Find all ``(start, end)`` vision-token spans in each batch element.

    Uses :func:`torch.where` for robust detection of every
    ``<|vision_start|>`` / ``<|vision_end|>`` pair, fixing the original
    ``list.index`` approach that only found the first occurrence and
    silently missed additional vision segments.

    Args:
        input_ids: ``(B, T)`` token-id tensor.
        vision_start_id: token id for ``<|vision_start|>``.
        vision_end_id: token id for ``<|vision_end|>``.

    Returns:
        A list (length ``B``) of lists of ``(start_pos, end_pos)`` tuples
        (inclusive on both ends).  An empty inner list means no vision
        tokens were found for that batch element.
    """
    B = input_ids.size(0)
    result: list[list[tuple[int, int]]] = []
    for b in range(B):
        row = input_ids[b]
        starts = torch.where(row == vision_start_id)[0]
        ends = torch.where(row == vision_end_id)[0]

        pairs: list[tuple[int, int]] = []
        ei = 0
        for s in starts:
            while ei < len(ends) and ends[ei] <= s:
                ei += 1
            if ei < len(ends):
                pairs.append((s.item(), ends[ei].item()))
                ei += 1
        result.append(pairs)
    return result


def _pairs_to_mask(
    pairs: list[list[tuple[int, int]]],
    shape: tuple[int, int],
    device: torch.device,
) -> torch.BoolTensor:
    """Convert ``select_visual_positions`` output to a dense boolean mask."""
    B, T = shape
    mask = torch.zeros(B, T, device=device, dtype=torch.bool)
    for b, segs in enumerate(pairs):
        for s, e in segs:
            mask[b, s : e + 1] = True
    return mask


# -----------------------------------------------------------------------
# CineMemModel
# -----------------------------------------------------------------------


class CineMemModel(nn.Module):
    """Memory-augmented wrapper around a Qwen2.5-VL backbone.

    Supports two memory formation modes selected by
    ``config.use_dual_memory``:

    * **Dual memory** (default): a shared
      :class:`~cardio.vlm.query_builder.QueryBuilder` feeds into
      TDM (transient dynamics) and PSM (persistent structure) pathways
      managed by :class:`~cardio.vlm.dual_memory.DualMemoryManager`.
    * **Legacy**: separate short-term and long-term formers (original
      CineMem design).

    Args:
        base_model: a ``transformers`` causal-LM (e.g. Qwen2.5-VL).
        tokenizer: the corresponding tokenizer with special tokens added.
        processor: the VL processor used for prompt encoding.
        config: full :class:`~cardio.vlm.config.CineMemConfig`.
        cinema_dim: if provided, a ``Linear(cinema_dim, hidden_size)``
            projection is created so that CineMA vision features can be
            injected via :meth:`_build_H`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        processor: Any,
        config: CineMemConfig,
        *,
        cinema_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.cfg = config

        # --- Hidden size detection ---
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(base_model.config, "text_config"):
            hidden_size = getattr(base_model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            msg = "Could not infer hidden size from base_model.config."
            raise ValueError(msg)
        self.hidden_size: int = hidden_size

        # --- Optional CineMA feature projection ---
        self.cinema_proj: nn.Linear | None = None
        if cinema_dim is not None:
            self.cinema_proj = nn.Linear(cinema_dim, hidden_size)

        # --- Memory system ---
        self.invocation_log: list[dict[str, Any]] = []

        if config.use_dual_memory:
            self._init_dual_memory(config, hidden_size, base_model)
        else:
            self._init_legacy_memory(config, hidden_size, base_model)

        # --- MRoPE temporal fix (Qwen2.5-VL issues #893 / #1031) ---
        self.temporal_override: TemporalAlignmentOverride | None = None
        if config.fix_mrope_temporal and hasattr(base_model, "get_rope_index"):
            self.temporal_override = TemporalAlignmentOverride(
                base_model,
                default_total_pixels=config.mrope_default_total_pixels,
            )
            self.temporal_override.patch_model()

        # --- Token IDs (always register legacy tokens) ---
        self.short_invoke_id = self._must_resolve_token(config.short_invoke_token)
        self.short_end_id = self._must_resolve_token(config.short_end_token)
        self.long_invoke_id = self._must_resolve_token(config.long_invoke_token)
        self.long_end_id = self._must_resolve_token(config.long_end_token)

        # Dual-memory token IDs (None when dual memory is disabled)
        if config.use_dual_memory:
            self.tdm_invoke_id: int | None = self._must_resolve_token(config.tdm_invoke_token)
            self.tdm_end_id: int | None = self._must_resolve_token(config.tdm_end_token)
            self.psm_invoke_id: int | None = self._must_resolve_token(config.psm_invoke_token)
            self.psm_end_id: int | None = self._must_resolve_token(config.psm_end_token)
        else:
            self.tdm_invoke_id = None
            self.tdm_end_id = None
            self.psm_invoke_id = None
            self.psm_end_id = None

        # Vision delimiter IDs (best-effort; None if tokenizer lacks them)
        self._vs_id = self._try_resolve_token("<|vision_start|>")
        self._ve_id = self._try_resolve_token("<|vision_end|>")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _must_resolve_token(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.tokenizer.unk_token_id:
            msg = f"Special token {token!r} not found in tokenizer. Call add_cinemem_tokens() first."
            raise ValueError(msg)
        return tid

    def _try_resolve_token(self, token: str) -> int | None:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.tokenizer.unk_token_id:
            return None
        return tid

    def _init_dual_memory(
        self,
        config: CineMemConfig,
        hidden_size: int,
        base_model: nn.Module,
    ) -> None:
        """Set up the TDM + PSM dual-memory pathway."""
        self.dual_memory = DualMemoryManager(
            config=config,
            hidden_size=hidden_size,
            base_model=base_model if config.former_backend == "lora_llm" else None,
        )
        # Legacy formers unused
        self.query_builder = None
        self.short_former = None
        self.long_former = None
        self.peft_model = None

    def _init_legacy_memory(
        self,
        config: CineMemConfig,
        hidden_size: int,
        base_model: nn.Module,
    ) -> None:
        """Set up legacy short / long memory formers."""
        self.dual_memory = None

        qb = config.query_builder
        self.query_builder = QueryBuilder(
            hidden_size=hidden_size,
            query_len=config.query_len,
            num_layers=qb.num_layers,
            num_heads=qb.num_heads,
            dropout=qb.dropout,
            ff_mult=qb.ff_mult,
        )

        self.former_backend = config.former_backend

        if self.former_backend == "tiny_transformer" or not is_peft_available():
            self.short_former = TinyMemoryFormer(hidden_size, config.short_mem_len)
            self.long_former = TinyMemoryFormer(hidden_size, config.long_mem_len)
            self.peft_model = None
        elif self.former_backend == "lora_llm":
            lora = config.lora
            short_targets = lora.short_target_modules or lora.target_modules
            long_targets = lora.long_target_modules or lora.target_modules

            self.peft_model = make_lora_adapters(
                base_model, "short_former",
                lora.r, lora.alpha, lora.dropout, short_targets,
            )
            from peft import LoraConfig as PeftLoraConfig

            self.peft_model.add_adapter(
                "long_former",
                PeftLoraConfig(
                    r=lora.r,
                    lora_alpha=lora.alpha,
                    lora_dropout=lora.dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=long_targets,
                ),
            )
            self.m_init_short = nn.Parameter(
                torch.randn(1, config.short_mem_len, hidden_size) * 0.02,
            )
            self.m_init_long = nn.Parameter(
                torch.randn(1, config.long_mem_len, hidden_size) * 0.02,
            )
            self.short_former = None
            self.long_former = None
        else:
            msg = f"Unknown former_backend: {self.former_backend!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Hidden-state construction
    # ------------------------------------------------------------------

    def _build_H(
        self,
        visual_states: torch.Tensor,
        text_states: torch.Tensor,
        cinema_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the hidden-state matrix for memory formation.

        Concatenates ``[visual_states, cinema_features_proj, text_states]``
        along the sequence dimension.

        Args:
            visual_states: ``(B, L_v, D)`` visual hidden states gathered
                from the backbone output.
            text_states: ``(B, L_t, D)`` text hidden states.
            cinema_features: optional ``(B, L_c, cinema_dim)`` features
                from a CineMA encoder (ConvViT / ConvUNetR).  Projected
                to ``hidden_size`` via :attr:`cinema_proj` before
                concatenation.  Ignored when ``cinema_proj`` is ``None``.

        Returns:
            ``(B, L_v + [L_c +] L_t', D)`` hidden-state tensor where
            ``L_t' = min(L_t, max_prompt_hidden)``.
        """
        if text_states.size(1) > self.cfg.max_prompt_hidden:
            text_states = text_states[:, -self.cfg.max_prompt_hidden :, :]

        parts: list[torch.Tensor] = [visual_states]

        if cinema_features is not None and self.cinema_proj is not None:
            parts.append(self.cinema_proj(cinema_features))

        parts.append(text_states)
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Memory formation
    # ------------------------------------------------------------------

    def form_memory(
        self,
        H: torch.Tensor,
        mem_type: str,
        *,
        n_frames: int = 2,
    ) -> torch.Tensor:
        """Form memory vectors, routing to the correct backend.

        Args:
            H: ``(B, L, D)`` combined hidden states.
            mem_type: one of ``"short"``, ``"long"``, ``"tdm"``, ``"psm"``.
            n_frames: temporal frame count (only used by dual-memory
                preprocessors).

        Returns:
            ``(B, N, D)`` memory tensor.
        """
        if self.cfg.use_dual_memory:
            assert self.dual_memory is not None  # noqa: S101
            dual_type = _MEM_TYPE_DUAL_MAP[mem_type]
            return self.dual_memory.form_memory(H, dual_type, n_frames=n_frames)

        return self._legacy_form_memory(H, _MEM_TYPE_LEGACY_MAP[mem_type])

    def _legacy_form_memory(self, H: torch.Tensor, mem_type: str) -> torch.Tensor:
        """Legacy short/long memory path."""
        assert self.query_builder is not None  # noqa: S101
        Q = self.query_builder(H)

        if self.peft_model is None:
            former = self.short_former if mem_type == "short" else self.long_former
            assert former is not None  # noqa: S101
            return former(H, Q)

        mem_len = self.cfg.short_mem_len if mem_type == "short" else self.cfg.long_mem_len
        adapter = "short_former" if mem_type == "short" else "long_former"
        return self._former_forward_lora(H, Q, mem_len, adapter)

    def _former_forward_lora(
        self,
        X: torch.Tensor,
        Q: torch.Tensor,
        mem_len: int,
        adapter_name: str,
    ) -> torch.Tensor:
        """Run memory formation through a LoRA adapter (legacy path)."""
        assert self.peft_model is not None  # noqa: S101
        set_active_adapter(self.peft_model, adapter_name)
        B = X.size(0)

        m_init = self.m_init_short if adapter_name == "short_former" else self.m_init_long
        m = m_init.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)

        inp = torch.cat([X, Q, m], dim=1)
        attn = torch.ones(B, inp.size(1), device=X.device, dtype=torch.long)

        out = self.peft_model(
            inputs_embeds=inp,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        return out.hidden_states[-1][:, -mem_len:, :]

    # ------------------------------------------------------------------
    # Visual projector passthrough (legacy short-memory only)
    # ------------------------------------------------------------------

    def _maybe_project_short_memory(self, M: torch.Tensor) -> torch.Tensor:
        """Project short memory through the VLM visual projector if present."""
        proj = (
            getattr(self.base_model, "visual_projector", None)
            or getattr(self.base_model, "vision_projector", None)
            or getattr(self.base_model, "multi_modal_projector", None)
        )
        if proj is None:
            return M
        try:
            return proj(M)
        except Exception:  # noqa: BLE001
            return M

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _gather_padded(
        self,
        states: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Gather marked positions into a dense, zero-padded tensor."""
        B, T, D = states.shape
        lens = mask.sum(dim=1)
        max_len = int(lens.max().item()) if lens.numel() else 0
        out = states.new_zeros((B, max(max_len, 1), D))
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[b, : idx.numel()] = states[b, idx]
        return out

    # ------------------------------------------------------------------
    # Token-type resolution for the generation loop
    # ------------------------------------------------------------------

    def _build_invoke_table(self) -> list[tuple[int, str, int]]:
        """Return ``(invoke_id, type_label, end_id)`` for every active invoke token."""
        table: list[tuple[int, str, int]] = [
            (self.short_invoke_id, "short", self.short_end_id),
            (self.long_invoke_id, "long", self.long_end_id),
        ]
        if self.tdm_invoke_id is not None and self.tdm_end_id is not None:
            table.append((self.tdm_invoke_id, "tdm", self.tdm_end_id))
        if self.psm_invoke_id is not None and self.psm_end_id is not None:
            table.append((self.psm_invoke_id, "psm", self.psm_end_id))
        return table

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: list[Any] | None,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_cinemem: bool = True,
        return_token_ids: bool = False,
        skip_special_tokens: bool = True,
        reverse_mem_type: bool = False,
        cinema_features: torch.Tensor | None = None,
        n_frames: int = 2,
    ) -> list[str] | tuple[list[str], torch.Tensor]:
        """Generate text with memory-augmented decoding.

        Args:
            images: list with a single image / frame-list (or ``None``).
            prompts: list with a single prompt string.
            max_new_tokens: maximum tokens to generate.
            temperature: sampling temperature (0 = greedy).
            top_p: nucleus-sampling threshold.
            enable_cinemem: whether to respond to memory invocation tokens.
            return_token_ids: if ``True``, also return the generated ids.
            skip_special_tokens: strip special tokens from decoded text.
            reverse_mem_type: swap short↔long for legacy tokens (debug).
            cinema_features: optional ``(1, L_c, cinema_dim)`` CineMA
                features injected into the hidden-state matrix.
            n_frames: number of temporal frames (for dual-memory
                preprocessors).

        Returns:
            Decoded text list; optionally also the ``(B, L_gen)`` id tensor.
        """
        self.invocation_log = []

        img = images[0] if images else None
        inputs = build_processor_inputs(self.processor, img, prompts[0])
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        # --- Prefill ---
        out = self.base_model(
            **inputs, use_cache=True, output_hidden_states=True,
        )
        past = out.past_key_values
        cur_logits = out.logits[:, -1, :]
        hidden_last = out.hidden_states[-1]  # (B, T, D)

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            msg = "Processor did not return input_ids; check your Qwen2.5-VL processor."
            raise ValueError(msg)
        B = input_ids.size(0)

        # --- Extract visual hidden states ---
        if self._vs_id is not None and self._ve_id is not None:
            vis_pairs = select_visual_positions(input_ids, self._vs_id, self._ve_id)
            vis_mask = _pairs_to_mask(vis_pairs, input_ids.shape, input_ids.device)
        else:
            vis_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        if vis_mask.any():
            visual_states = self._gather_padded(hidden_last, vis_mask)
        else:
            visual_states = torch.zeros(
                B, 0, self.hidden_size, device=self.device, dtype=hidden_last.dtype,
            )

        seg_hiddens: list[torch.Tensor] = []
        generated: list[torch.Tensor] = []

        invoke_table = self._build_invoke_table() if enable_cinemem else []

        # --- Autoregressive decode ---
        for step in range(max_new_tokens):
            next_id = self._sample_next(cur_logits, temperature, top_p)
            generated.append(next_id)
            seg_hiddens.append(hidden_last[:, -1:, :])

            # Check for memory invocation tokens
            matched = self._match_invoke(next_id, invoke_table)

            if matched is not None:
                tok_type, end_id = matched

                self.invocation_log.append({
                    "step": step,
                    "type": tok_type,
                    "token_idx": int(next_id[0].item()),
                })

                # Feed invocation token through the backbone
                out = self.base_model(
                    input_ids=next_id.unsqueeze(-1),
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]
                seg_hiddens.append(hidden_last)

                # Resolve effective mem_type (legacy reversal only)
                mem_type = self._resolve_mem_type(tok_type, reverse_mem_type)

                # Build H and form memory
                text_states = torch.cat(seg_hiddens, dim=1)
                H = self._build_H(visual_states, text_states, cinema_features)
                M = self.form_memory(H, mem_type, n_frames=n_frames)

                if mem_type == "short":
                    M = self._maybe_project_short_memory(M)

                # Insert memory embeddings
                out = self.base_model(
                    inputs_embeds=M,
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]

                # Emit end token
                end_tensor = torch.full((B,), end_id, device=self.device, dtype=torch.long)
                generated.append(end_tensor)

                out = self.base_model(
                    input_ids=end_tensor.unsqueeze(-1),
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]
                seg_hiddens = []
                cur_logits = out.logits[:, -1, :]
                continue

            # Normal decoding step
            out = self.base_model(
                input_ids=next_id.unsqueeze(-1),
                use_cache=True,
                past_key_values=past,
                output_hidden_states=True,
            )
            past = out.past_key_values
            hidden_last = out.hidden_states[-1]
            cur_logits = out.logits[:, -1, :]

            if (next_id == self.tokenizer.eos_token_id).all():
                break

        gen_ids = torch.stack(generated, dim=1)
        texts = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=skip_special_tokens,
        )
        if return_token_ids:
            return texts, gen_ids
        return texts

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Sample (or argmax) the next token from *logits*."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            drop = cum > top_p
            drop[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(drop, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            chosen = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _match_invoke(
        next_id: torch.Tensor,
        table: list[tuple[int, str, int]],
    ) -> tuple[str, int] | None:
        """Check whether *next_id* matches any invoke token in *table*."""
        for invoke_id, tok_type, end_id in table:
            if (next_id == invoke_id).any():
                return tok_type, end_id
        return None

    @staticmethod
    def _resolve_mem_type(tok_type: str, reverse: bool) -> str:
        """Apply the legacy ``reverse_mem_type`` swap (short↔long only)."""
        if not reverse:
            return tok_type
        if tok_type == "short":
            return "long"
        if tok_type == "long":
            return "short"
        return tok_type
