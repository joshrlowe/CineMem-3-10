"""Lightweight memory former (standalone transformer backend).

Used as the ``tiny_transformer`` backend for memory formation.  The input
is the concatenation ``[X, Q, m_init]`` where:

- ``X`` is the (possibly pre-processed) hidden-state context,
- ``Q`` is the query output from :class:`~cardio.vlm.query_builder.QueryBuilder`,
- ``m_init`` is a learnable memory initialisation.

The last ``mem_len`` positions of the encoder output are returned as the
compressed memory ``M``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyMemoryFormer(nn.Module):
    """Small transformer encoder that compresses context into memory vectors.

    Args:
        hidden_size: model hidden dimension ``D``.
        mem_len: number of memory vectors ``N`` to produce.
        num_layers: transformer encoder depth.
        num_heads: number of attention heads.
        dropout: attention / feed-forward dropout.
        ff_mult: feed-forward expansion factor.
    """

    def __init__(
        self,
        hidden_size: int,
        mem_len: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.mem_len = mem_len
        self.m_init = nn.Parameter(torch.randn(mem_len, hidden_size) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Compress ``(X, Q)`` into ``mem_len`` memory vectors.

        Args:
            X: ``(B, L_x, D)`` context hidden states.
            Q: ``(B, K, D)`` query vectors from the QueryBuilder.

        Returns:
            ``(B, N, D)`` memory tensor where ``N = mem_len``.
        """
        B = X.size(0)
        m = self.m_init.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([X, Q, m], dim=1)
        out = self.ln(self.encoder(inp))
        return out[:, -self.mem_len :, :]
