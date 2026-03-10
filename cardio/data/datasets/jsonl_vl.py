"""JSONL-based vision-language dataset for cardiac VQA."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset


@dataclass
class Sample:
    id: str
    image: str | None
    prompt: str
    answer: str | None = None
    meta: dict[str, Any] | None = None


class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path: str, image_dir: str = "") -> None:
        self.items: list[Sample] = []
        self.image_dir = image_dir
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_image = obj.get("image", None)
                if raw_image is not None:
                    if os.path.isabs(raw_image):
                        image_path = raw_image
                    else:
                        image_path = self.image_dir + "/" + obj.get("dataset", "") + "/" + raw_image
                else:
                    image_path = None
                self.items.append(
                    Sample(
                        id=str(obj.get("id", len(self.items))),
                        image=image_path,
                        prompt=obj["prompt"],
                        answer=obj.get("answer", None),
                        meta={k: v for k, v in obj.items() if k not in ("id", "image", "prompt", "answer")},
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]
