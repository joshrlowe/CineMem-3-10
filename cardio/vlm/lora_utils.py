"""LoRA adapter management utilities.

Thin wrappers around PEFT for creating and switching LoRA adapters on a
causal-LM backbone.
"""

from __future__ import annotations


def is_peft_available() -> bool:
    """Return ``True`` if the ``peft`` library is importable."""
    try:
        import peft  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def make_lora_adapters(
    base_model,
    adapter_name: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
):
    """Wrap *base_model* with a named LoRA adapter via PEFT.

    Args:
        base_model: a ``transformers`` model instance.
        adapter_name: name tag for the adapter (e.g. ``"short"``).
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: dropout applied to LoRA layers.
        target_modules: list of module name patterns to inject into.

    Returns:
        A ``PeftModel`` wrapping *base_model*.
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(base_model, lora_cfg, adapter_name=adapter_name)


def set_active_adapter(model, name: str) -> None:
    """Switch the active LoRA adapter on *model*.

    Compatible with ``PeftModel`` and multi-adapter wrapper APIs.
    """
    if hasattr(model, "set_adapter"):
        model.set_adapter(name)
    elif hasattr(model, "active_adapter"):
        model.active_adapter = name
    else:
        msg = "Model doesn't support adapter switching."
        raise AttributeError(msg)
