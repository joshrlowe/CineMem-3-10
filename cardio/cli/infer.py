r"""Interactive inference CLI for CineMemModel.

Ported from ``_reference/CineMem/main/cli/infer.py`` with extensions:

* ``--use-tools``: enables :class:`~cardio.tools.ToolRouter` to parse and
  execute ``<tool_call>`` blocks in generated text.
* ``--disable-memory``: sets ``enable_cinemem=False`` for ablation studies.

Usage::

    cardio-infer --config configs/cinemem_qwen25vl7b.yaml \
                 --prompt "Describe the cardiac function." \
                 --image /path/to/cine.nii.gz \
                 --use-tools
"""

from __future__ import annotations

import argparse
import json
import sys

from cardio.cli.common import build_cinemem_config, load_yaml
from cardio.data.collate import load_image
from cardio.utils.logging import get_logger
from cardio.utils.misc import to_torch_dtype
from cardio.vlm.model import CineMemModel
from cardio.vlm.qwen_loader import load_qwen25vl

logger = get_logger(__name__)

_w = sys.stdout.write


def _process_tool_calls(
    raw_output: str,
    *,
    router: object,
    context: dict,
) -> str:
    """Parse ``<tool_call>`` blocks, execute them, and inject results.

    Args:
        raw_output: model-generated text (may contain tool calls).
        router: :class:`~cardio.tools.ToolRouter` instance.
        context: runtime context passed to the router.

    Returns:
        The original output with ``<tool_result>...</tool_result>`` blocks
        appended after each executed tool call.
    """
    from cardio.tools import ToolRouter

    calls = ToolRouter.parse_tool_calls(raw_output)
    if not calls:
        return raw_output

    augmented = raw_output
    for call in calls:
        result = router.execute(call, context)
        result_json = json.dumps(result, default=str)
        augmented += f"\n<tool_result>{result_json}</tool_result>"
        logger.info("Tool %s -> %s", call["name"], result_json[:200])

    return augmented


def main() -> None:
    """Entry point for ``cardio-infer``."""
    ap = argparse.ArgumentParser(description="Run CineMemModel inference.")
    ap.add_argument("--config", default="configs/cinemem_qwen25vl7b.yaml", help="YAML config path.")
    ap.add_argument("--model_name_or_path", default=None, help="Override model name/path.")
    ap.add_argument("--image", default=None, help="Path to an image or NIfTI file.")
    ap.add_argument("--prompt", required=True, help="Text prompt.")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    ap.add_argument("--enable_cinemem", action="store_true", help="Enable CineMem memory modules.")
    ap.add_argument(
        "--disable-memory", action="store_true",
        help="Disable CineMem memory modules (ablation). Overrides --enable_cinemem.",
    )
    ap.add_argument(
        "--use-tools", action="store_true",
        help="Enable ToolRouter to execute <tool_call> blocks.",
    )
    args = ap.parse_args()

    # ----- config ---------------------------------------------------------
    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path

    viscfg = build_cinemem_config(cfg_dict)
    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    # ----- model ----------------------------------------------------------
    base_model, tokenizer, processor = load_qwen25vl(
        model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust,
    )
    cinemem = CineMemModel(base_model, tokenizer, processor, viscfg)
    cinemem.eval()

    # ----- tool router (lazy) ---------------------------------------------
    tool_router = None
    if args.use_tools:
        from cardio.tools import ToolRouter

        tool_router = ToolRouter()
        logger.info("ToolRouter enabled.")

    # ----- memory flag ----------------------------------------------------
    enable_cinemem = args.enable_cinemem
    if args.disable_memory:
        enable_cinemem = False
        logger.info("Memory modules disabled (ablation mode).")

    # ----- inference ------------------------------------------------------
    img = load_image(args.image)
    output = cinemem.generate(
        images=[img] if img else [None],
        prompts=[args.prompt],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_cinemem=enable_cinemem,
    )
    raw_text = output[0]

    if tool_router is not None:
        context = {"nifti_path": args.image} if args.image else {}
        raw_text = _process_tool_calls(raw_text, router=tool_router, context=context)

    _w(raw_text + "\n")


if __name__ == "__main__":
    main()
