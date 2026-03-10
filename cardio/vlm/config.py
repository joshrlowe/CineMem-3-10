"""CineMem / CardioVLM model configuration dataclasses.

Ported from ``_reference/CineMem/main/model/configuration_cinemem.py``
and ``_reference/CineMem/main/cli/common.py``, with the following fixes:

- **YAML key fix**: ``build_cinemem_config`` now reads from
  ``cfg_dict.get("cinemem", {})`` instead of ``cfg_dict.get("main", {})``.
- **Field typo fix**: ``long_invoke_token`` was being populated from
  ``v.get("long_end_token", ...)`` — corrected to ``v.get("long_invoke_token", ...)``.
- **Dual-memory extension**: ``CineMemConfig`` gains ``tdm_mem_len``,
  ``psm_mem_len``, ``use_dual_memory``, and the four TDM/PSM token fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cardio.vlm.constants import (
    LONG_END,
    LONG_INVOKE,
    PSM_END,
    PSM_INVOKE,
    SHORT_END,
    SHORT_INVOKE,
    TDM_END,
    TDM_INVOKE,
)


@dataclass
class QueryBuilderConfig:
    """Configuration for the memory query builder transformer."""

    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    ff_mult: int = 4


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter injection."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    short_target_modules: list[str] | None = None
    long_target_modules: list[str] | None = None


@dataclass
class CineMemConfig:
    """Full configuration for the CineMem / CardioVLM memory-augmented model."""

    # Legacy STM/LTM tokens
    short_invoke_token: str = SHORT_INVOKE
    short_end_token: str = SHORT_END
    long_invoke_token: str = LONG_INVOKE
    long_end_token: str = LONG_END

    # Dual-memory TDM/PSM tokens
    tdm_invoke_token: str = TDM_INVOKE
    tdm_end_token: str = TDM_END
    psm_invoke_token: str = PSM_INVOKE
    psm_end_token: str = PSM_END

    query_len: int = 8
    short_mem_len: int = 8
    long_mem_len: int = 16
    tdm_mem_len: int = 8
    psm_mem_len: int = 16
    use_dual_memory: bool = True

    query_builder: QueryBuilderConfig = field(default_factory=QueryBuilderConfig)

    former_backend: str = "lora_llm"  # lora_llm | tiny_transformer
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    max_prompt_hidden: int = 1024

    # MRoPE temporal position fix (Qwen2.5-VL issues #893 / #1031)
    fix_mrope_temporal: bool = True
    mrope_default_total_pixels: int = 1003520


def build_cinemem_config(cfg_dict: dict[str, Any]) -> CineMemConfig:
    """Build a ``CineMemConfig`` from a parsed YAML dict.

    Expects the YAML to have a top-level ``cinemem:`` key containing
    the memory-model configuration.  Missing keys fall back to dataclass
    defaults.
    """
    v = cfg_dict.get("cinemem", {})
    qb = v.get("query_builder", {})
    lora = v.get("lora", {})

    return CineMemConfig(
        short_invoke_token=v.get("short_invoke_token", SHORT_INVOKE),
        short_end_token=v.get("short_end_token", SHORT_END),
        long_invoke_token=v.get("long_invoke_token", LONG_INVOKE),
        long_end_token=v.get("long_end_token", LONG_END),
        tdm_invoke_token=v.get("tdm_invoke_token", TDM_INVOKE),
        tdm_end_token=v.get("tdm_end_token", TDM_END),
        psm_invoke_token=v.get("psm_invoke_token", PSM_INVOKE),
        psm_end_token=v.get("psm_end_token", PSM_END),
        query_len=int(v.get("query_len", 8)),
        short_mem_len=int(v.get("short_mem_len", 8)),
        long_mem_len=int(v.get("long_mem_len", 16)),
        tdm_mem_len=int(v.get("tdm_mem_len", 8)),
        psm_mem_len=int(v.get("psm_mem_len", 16)),
        use_dual_memory=bool(v.get("use_dual_memory", True)),
        former_backend=str(v.get("former_backend", "lora_llm")),
        max_prompt_hidden=int(v.get("max_prompt_hidden", 1024)),
        fix_mrope_temporal=bool(v.get("fix_mrope_temporal", True)),
        mrope_default_total_pixels=int(v.get("mrope_default_total_pixels", 1003520)),
        query_builder=QueryBuilderConfig(
            num_layers=int(qb.get("num_layers", 2)),
            num_heads=int(qb.get("num_heads", 8)),
            dropout=float(qb.get("dropout", 0.0)),
            ff_mult=int(qb.get("ff_mult", 4)),
        ),
        lora=LoRAConfig(
            r=int(lora.get("r", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(
                lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            ),
            short_target_modules=lora.get("short_target_modules"),
            long_target_modules=lora.get("long_target_modules"),
        ),
    )
