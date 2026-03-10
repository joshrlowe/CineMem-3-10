"""Mem4D-inspired dual memory system (TDM + PSM).

Separates memory formation into two complementary pathways:

* **Transient Dynamics Memory (TDM)** — captures high-frequency motion
  details from recent frames via temporal difference attention.  Used for
  wall-motion assessment, contraction patterns, and EF estimation.
  Invoked by the ``<tdm_I>`` token during generation.

* **Persistent Structure Memory (PSM)** — compresses and preserves
  long-term spatial information via exponential moving-average pooling
  across the temporal dimension.  Used for chamber geometry, structural
  abnormalities, and static measurements.  Invoked by the ``<psm_I>``
  token during generation.

Both modules accept the same :class:`~cardio.vlm.query_builder.QueryBuilder`
output as input and use the same MemoryFormer backend (LoRA or
``tiny_transformer``) but with separate adapter names (``"tdm_former"``
and ``"psm_former"``).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from cardio.vlm.config import CineMemConfig
from cardio.vlm.lora_utils import is_peft_available, make_lora_adapters, set_active_adapter
from cardio.vlm.memory_former import TinyMemoryFormer
from cardio.vlm.query_builder import QueryBuilder


# ---------------------------------------------------------------------------
# Transient Dynamics Memory
# ---------------------------------------------------------------------------


class TransientDynamicsMemory(nn.Module):
    """Capture high-frequency motion details from recent frames.

    Preprocessing: computes frame-to-frame deltas from the visual hidden
    states, then applies multi-head self-attention over the delta sequence
    so that cross-frame motion patterns can interact.

    The attended deltas are then fed (together with the shared query ``Q``)
    to the memory former backend to produce ``mem_len`` memory vectors.

    Args:
        hidden_size: model hidden dimension ``D``.
        mem_len: number of output memory vectors.
        num_heads: attention heads for the temporal-difference layer.
        dropout: dropout rate for the temporal-difference layer.
        use_tiny_former: if ``True``, create a standalone
            :class:`TinyMemoryFormer`; otherwise allocate a learnable
            ``m_init`` for the LoRA-LLM path.
    """

    def __init__(
        self,
        hidden_size: int,
        mem_len: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_tiny_former: bool = True,
    ) -> None:
        super().__init__()
        self.mem_len = mem_len
        self.hidden_size = hidden_size

        self.temporal_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(hidden_size)

        if use_tiny_former:
            self.former = TinyMemoryFormer(hidden_size, mem_len)
        else:
            self.former = None  # type: ignore[assignment]
            self.m_init = nn.Parameter(torch.randn(1, mem_len, hidden_size) * 0.02)

    def preprocess(self, H_visual: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Compute temporal-difference attention over frame deltas.

        Args:
            H_visual: ``(B, L_total, D)`` visual hidden states whose
                first dimension encodes ``n_frames`` consecutive frames,
                each with ``L_total // n_frames`` spatial tokens.
            n_frames: number of temporal frames encoded in *H_visual*.

        Returns:
            ``(B, L', D)`` attended delta features.  When ``n_frames < 2``
            the input is returned unchanged (no deltas to compute).
        """
        B, L_total, D = H_visual.shape
        if n_frames < 2:
            return H_visual

        L_per_frame = L_total // n_frames
        frames = H_visual[:, : n_frames * L_per_frame, :].reshape(B, n_frames, L_per_frame, D)

        delta = frames[:, 1:] - frames[:, :-1]  # (B, T-1, L, D)
        delta_flat = delta.reshape(B, -1, D)  # (B, (T-1)*L, D)

        attn_out, _ = self.temporal_attn(delta_flat, delta_flat, delta_flat)
        return self.temporal_ln(attn_out + delta_flat)


# ---------------------------------------------------------------------------
# Persistent Structure Memory
# ---------------------------------------------------------------------------


class PersistentStructureMemory(nn.Module):
    """Compress and preserve long-term spatial information.

    Preprocessing: applies an exponential moving average (EMA) with a
    learnable per-channel decay across the temporal dimension.  This
    suppresses transient motion and extracts stable geometric features.

    The pooled representation is then fed (together with the shared
    query ``Q``) to the memory former backend to produce ``mem_len``
    memory vectors.

    Args:
        hidden_size: model hidden dimension ``D``.
        mem_len: number of output memory vectors.
        use_tiny_former: if ``True``, create a standalone
            :class:`TinyMemoryFormer`; otherwise allocate a learnable
            ``m_init`` for the LoRA-LLM path.
    """

    def __init__(
        self,
        hidden_size: int,
        mem_len: int,
        *,
        use_tiny_former: bool = True,
    ) -> None:
        super().__init__()
        self.mem_len = mem_len
        self.hidden_size = hidden_size

        # Learnable per-channel decay (sigmoid maps to [0, 1])
        self.log_decay = nn.Parameter(torch.zeros(hidden_size))
        self.ema_ln = nn.LayerNorm(hidden_size)

        if use_tiny_former:
            self.former = TinyMemoryFormer(hidden_size, mem_len)
        else:
            self.former = None  # type: ignore[assignment]
            self.m_init = nn.Parameter(torch.randn(1, mem_len, hidden_size) * 0.02)

    def preprocess(self, H_visual: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Apply EMA pooling across the temporal dimension.

        Args:
            H_visual: ``(B, L_total, D)`` visual hidden states.
            n_frames: number of temporal frames encoded in *H_visual*.

        Returns:
            ``(B, L_per_frame, D)`` stable geometric features.  When
            ``n_frames < 2`` a layer-normed copy of the input is returned.
        """
        B, L_total, D = H_visual.shape
        if n_frames < 2:
            return self.ema_ln(H_visual)

        L_per_frame = L_total // n_frames
        frames = H_visual[:, : n_frames * L_per_frame, :].reshape(B, n_frames, L_per_frame, D)

        decay = torch.sigmoid(self.log_decay)  # (D,)

        ema = frames[:, 0]  # (B, L, D)
        for t in range(1, n_frames):
            ema = decay * ema + (1.0 - decay) * frames[:, t]

        return self.ema_ln(ema)  # (B, L_per_frame, D)


# ---------------------------------------------------------------------------
# Dual Memory Manager
# ---------------------------------------------------------------------------


class DualMemoryManager(nn.Module):
    """Orchestrate TDM and PSM queries during generation.

    Holds a shared :class:`QueryBuilder`, a :class:`TransientDynamicsMemory`,
    and a :class:`PersistentStructureMemory`.  The :meth:`form_memory` method
    computes a shared query ``Q`` and routes to the appropriate memory
    module based on the invocation token type.

    For the ``lora_llm`` backend, two LoRA adapters (``"tdm_former"`` and
    ``"psm_former"``) are registered on the provided *base_model*.  For the
    ``tiny_transformer`` backend, each memory module uses its own standalone
    :class:`TinyMemoryFormer`.

    Args:
        config: full model configuration.
        hidden_size: model hidden dimension ``D``.
        base_model: the Qwen backbone (required for ``lora_llm`` backend).
    """

    def __init__(
        self,
        config: CineMemConfig,
        hidden_size: int,
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.hidden_size = hidden_size

        # Shared query builder
        qb = config.query_builder
        self.query_builder = QueryBuilder(
            hidden_size=hidden_size,
            query_len=config.query_len,
            num_layers=qb.num_layers,
            num_heads=qb.num_heads,
            dropout=qb.dropout,
            ff_mult=qb.ff_mult,
        )

        # Backend selection
        use_tiny = config.former_backend == "tiny_transformer" or not is_peft_available()

        self.tdm = TransientDynamicsMemory(
            hidden_size, config.tdm_mem_len, use_tiny_former=use_tiny,
        )
        self.psm = PersistentStructureMemory(
            hidden_size, config.psm_mem_len, use_tiny_former=use_tiny,
        )

        self.peft_model: nn.Module | None = None

        if not use_tiny:
            if base_model is None:
                msg = "base_model is required for the lora_llm backend."
                raise ValueError(msg)

            lora = config.lora
            targets = lora.target_modules

            self.peft_model = make_lora_adapters(
                base_model, "tdm_former", lora.r, lora.alpha, lora.dropout, targets,
            )

            from peft import LoraConfig as PeftLoraConfig

            self.peft_model.add_adapter(
                "psm_former",
                PeftLoraConfig(
                    r=lora.r,
                    lora_alpha=lora.alpha,
                    lora_dropout=lora.dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=targets,
                ),
            )

    # -----------------------------------------------------------------
    # LoRA-LLM forward (mirrors CineMem's _former_forward_lora)
    # -----------------------------------------------------------------

    def _former_forward_lora(
        self,
        X: torch.Tensor,
        Q: torch.Tensor,
        mem_len: int,
        adapter_name: str,
        m_init: nn.Parameter,
    ) -> torch.Tensor:
        """Run memory formation through a LoRA adapter on the LLM backbone.

        Switches to *adapter_name*, concatenates ``[X, Q, m_init]`` as
        ``inputs_embeds``, and extracts the last *mem_len* hidden states
        from the final layer.
        """
        assert self.peft_model is not None  # noqa: S101
        set_active_adapter(self.peft_model, adapter_name)
        B = X.size(0)

        m = m_init.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)
        inp = torch.cat([X, Q, m], dim=1)
        attn = torch.ones(B, inp.size(1), device=X.device, dtype=torch.long)

        out = self.peft_model(
            inputs_embeds=inp,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        hs = out.hidden_states[-1]
        return hs[:, -mem_len:, :]

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def form_memory(
        self,
        H: torch.Tensor,
        mem_type: Literal["tdm", "psm"],
        *,
        n_frames: int = 2,
    ) -> torch.Tensor:
        """Form memory vectors from hidden states.

        Args:
            H: ``(B, L, D)`` combined visual + text hidden states.
            mem_type: ``"tdm"`` for transient dynamics or ``"psm"`` for
                persistent structure.
            n_frames: number of temporal frames encoded in the visual
                portion of *H* (used by the TDM/PSM preprocessors).

        Returns:
            ``(B, N, D)`` memory tensor (``N = tdm_mem_len`` or
            ``psm_mem_len`` depending on *mem_type*).
        """
        Q = self.query_builder(H)

        if mem_type == "tdm":
            X = self.tdm.preprocess(H, n_frames)
            if self.peft_model is None:
                return self.tdm.former(X, Q)
            return self._former_forward_lora(
                X, Q, self.tdm.mem_len, "tdm_former", self.tdm.m_init,
            )

        if mem_type == "psm":
            X = self.psm.preprocess(H, n_frames)
            if self.peft_model is None:
                return self.psm.former(X, Q)
            return self._former_forward_lora(
                X, Q, self.psm.mem_len, "psm_former", self.psm.m_init,
            )

        msg = f"Unknown mem_type: {mem_type!r}. Use 'tdm' or 'psm'."
        raise ValueError(msg)
