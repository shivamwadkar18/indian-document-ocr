"""Torch dataset and collate function for the JSONL recognition splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from idocr.data.recognition.dataset import ANNOTATIONS_FILENAME
from idocr.data.recognition.schema import RecognitionSample
from idocr.models.recognizer.crnn import images_to_tensor
from idocr.models.recognizer.vocab import Vocabulary


def read_split(split_dir: str | Path, limit: int | None = None) -> list[RecognitionSample]:
    """Read ``annotations.jsonl`` of one split (first ``limit`` records)."""
    path = Path(split_dir) / ANNOTATIONS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"missing annotations: {path}")
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(samples) >= limit:
                break
            if line.strip():
                samples.append(RecognitionSample(**json.loads(line)))
    if not samples:
        raise ValueError(f"no records in {path}")
    return samples


class RecognitionDataset(Dataset):
    """One split; items are ``(grayscale PIL image, label indices, sample)``."""

    def __init__(
        self,
        split_dir: str | Path,
        vocab: Vocabulary,
        *,
        image_height: int,
        limit: int | None = None,
        augment: Callable[[Image.Image, np.random.Generator], Image.Image] | None = None,
        seed: int = 0,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.vocab = vocab
        self.image_height = image_height
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.samples = read_split(self.split_dir, limit)
        # Fail fast on characters the vocabulary cannot represent.
        self.targets = [vocab.encode(s.text) for s in self.samples]

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw; same (seed, epoch, index) -> same image."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, list[int], RecognitionSample]:
        sample = self.samples[index]
        with Image.open(self.split_dir / sample.image_path) as im:
            image = im.convert("L")
        if image.height != self.image_height:
            raise ValueError(
                f"{sample.image_path}: height {image.height} != {self.image_height}"
            )
        if self.augment is not None:
            image = self.augment(image, np.random.default_rng([self.seed, self.epoch, index]))
        return image, self.targets[index], sample


def collate_batch(items: list[tuple[Image.Image, list[int], RecognitionSample]]) -> dict[str, Any]:
    """Pad images to the batch max width and concatenate CTC targets.

    Returns ``images (B,1,H,W)``, ``widths (B,)``, ``targets (sum L,)``,
    ``target_lengths (B,)``, ``texts`` and ``fields``.
    """
    images, targets, samples = zip(*items)
    batch, widths = images_to_tensor(images)
    return {
        "images": batch,
        "widths": widths,
        "targets": torch.tensor([i for t in targets for i in t], dtype=torch.long),
        "target_lengths": torch.tensor([len(t) for t in targets], dtype=torch.long),
        "texts": [s.text for s in samples],
        "fields": [s.field_name for s in samples],
    }
