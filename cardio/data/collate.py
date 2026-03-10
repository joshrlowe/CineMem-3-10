"""Collation utilities for vision-language batches."""

from __future__ import annotations

from typing import Any

from PIL import Image

from cardio.data.datasets.jsonl_vl import Sample
from cardio.data.io.nifti import load_nifti_as_pil


def load_image(path: str | None) -> Image.Image | list[Image.Image] | None:
    """Load an image or NIfTI volume, returning PIL images.

    For NIfTI files (4-D cine), returns a list of frames; for 3-D/2-D NIfTI or
    standard image files, returns a single ``PIL.Image``.
    """
    if path is None:
        return None
    if path.endswith((".nii", ".nii.gz")):
        frames = load_nifti_as_pil(path)
        return frames if len(frames) > 1 else frames[0]
    return Image.open(path).convert("RGB")


def collate_samples(samples: list[Sample]) -> dict[str, Any]:
    """Collate a list of :class:`Sample` into a batch dictionary."""
    ids = [s.id for s in samples]
    images = [load_image(s.image) for s in samples]
    prompts = [s.prompt for s in samples]
    answers = [s.answer for s in samples]
    return {"ids": ids, "images": images, "prompts": prompts, "answers": answers}


def build_processor_inputs(
    processor: Any,
    image_or_frames: Image.Image | list[Image.Image] | None,
    prompt: str,
) -> dict[str, Any]:
    """Build Qwen2.5-VL processor inputs, routing cine frame-lists to *videos*."""
    content: list[dict[str, Any]] = []
    images_kw: list[Image.Image] | None = None
    videos_kw: list[list[Image.Image]] | None = None

    if image_or_frames is not None:
        if isinstance(image_or_frames, list):
            content.append({"type": "video", "video": image_or_frames})
            videos_kw = [image_or_frames]
        else:
            content.append({"type": "image", "image": image_or_frames})
            images_kw = [image_or_frames]

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
    if images_kw is not None:
        kwargs["images"] = images_kw
    if videos_kw is not None:
        kwargs["videos"] = videos_kw
    return processor(**kwargs)
