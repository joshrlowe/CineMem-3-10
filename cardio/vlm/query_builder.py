"""Learnable query builder for memory formation.

Appends learnable query tokens to the input hidden states and passes
the concatenation through a transformer encoder.  A causal mask prevents
the original hidden-state positions from attending to the query positions,
so the input representation is not distorted.  The final ``query_len``
positions of the output are returned as the query ``Q``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryBuilder(nn.Module):
    """Build a fixed-length query from variable-length hidden states.

    Args:
        hidden_size: model hidden dimension ``D``.
        query_len: number of query vectors ``K`` to produce.
        num_layers: transformer encoder depth.
        num_heads: number of attention heads.
        dropout: attention / feed-forward dropout.
        ff_mult: feed-forward expansion factor.
        max_len: maximum supported sequence length (``L + K``).
    """

    def __init__(
        self,
        hidden_size: int,
        query_len: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_mult: int = 4,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_len = query_len
        self.q_init = nn.Parameter(torch.randn(query_len, hidden_size) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        H: torch.Tensor,
        H_key_padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Produce query vectors from hidden states.

        Args:
            H: ``(B, L, D)`` hidden-state sequence.
            H_key_padding_mask: ``(B, L)`` boolean mask (``True`` = padding).

        Returns:
            ``(B, K, D)`` query tensor where ``K = query_len``.
        """
        B, L, D = H.shape
        q = self.q_init.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        x = torch.cat([H, q], dim=1)  # (B, L+K, D)

        if x.size(1) > self.pos_emb.size(1):
            msg = f"Sequence too long for pos_emb: {x.size(1)} > {self.pos_emb.size(1)}. Increase max_len."
            raise ValueError(msg)
        x = x + self.pos_emb[:, : x.size(1), :]

        # Causal mask: H tokens cannot attend to Q positions
        total = L + self.query_len
        attn_mask = torch.zeros(total, total, device=H.device, dtype=torch.float32)
        attn_mask[:L, L:] = float("-inf")

        src_kpm = None
        if H_key_padding_mask is not None:
            q_kpm = torch.zeros((B, self.query_len), device=H.device, dtype=torch.bool)
            src_kpm = torch.cat([H_key_padding_mask, q_kpm], dim=1)

        out = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_kpm)
        out = self.ln(out)
        return out[:, -self.query_len :, :]
