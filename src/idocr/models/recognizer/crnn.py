"""CRNN text-line recognizer (CNN -> BiLSTM -> CTC) and its TextRecognizer adapter.

Input: grayscale line images of fixed height (``target_height + 2 *
vertical_padding`` = 56 by default) and variable width, as produced by
:func:`idocr.data.preprocessing.ocr.preprocess_ocr_crop`. The CNN downsamples
width by 4, so a crop of width ``W`` yields ``W // 4`` CTC frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
    preprocess_ocr_crop,
)
from idocr.models.recognizer.base import TextRecognizer
from idocr.models.recognizer.vocab import DEFAULT_ALPHABET, Vocabulary
from idocr.types import ImageArray, RecognitionResult

WIDTH_DOWNSAMPLE = 4
PAD_VALUE = 255  # same white as preprocess_ocr_crop padding
# White margin added right of every image in a batch. It exceeds the CNN's
# one-sided receptive field (~17 px), so a crop's frames are identical whether
# it is run alone or padded next to wider crops.
MIN_RIGHT_PAD = 20


@dataclass(frozen=True)
class CRNNConfig:
    num_classes: int = len(DEFAULT_ALPHABET) + 1
    image_height: int = DEFAULT_TARGET_HEIGHT + 2 * DEFAULT_VERTICAL_PADDING
    cnn_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1


def _conv(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """Small CNN + BiLSTM + linear head; returns CTC log-probs ``(T, B, C)``."""

    def __init__(self, config: CRNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CRNNConfig()
        c1, c2, c3, c4 = self.config.cnn_channels
        self.cnn = nn.Sequential(
            *_conv(1, c1), nn.MaxPool2d(2, 2),            # H/2, W/2
            *_conv(c1, c2), nn.MaxPool2d(2, 2),           # H/4, W/4
            *_conv(c2, c3), *_conv(c3, c3),
            nn.MaxPool2d((2, 1), (2, 1)),                 # H/8, W/4
            *_conv(c3, c4),
            nn.MaxPool2d((2, 1), (2, 1)),                 # H/16, W/4
            nn.AdaptiveAvgPool2d((1, None)),              # collapse height
        )
        self.rnn = nn.LSTM(
            c4,
            self.config.hidden_size,
            num_layers=self.config.lstm_layers,
            bidirectional=True,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * self.config.hidden_size, self.config.num_classes)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, WIDTH_DOWNSAMPLE, rounding_mode="floor")

    def forward(self, images: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """``images``: ``(B, 1, H, W)`` float in [-1, 1]; ``widths``: true widths ``(B,)``.

        Returns float32 log-probabilities ``(T, B, num_classes)``.
        """
        features = self.cnn(images).squeeze(2).permute(2, 0, 1)  # (T, B, C)
        if widths is not None:
            lengths = self.output_lengths(widths).clamp(min=1, max=features.size(0))
            packed = pack_padded_sequence(features, lengths.cpu(), enforce_sorted=False)
            seq, _ = pad_packed_sequence(self.rnn(packed)[0], total_length=features.size(0))
        else:
            seq, _ = self.rnn(features)
        return self.classifier(seq).float().log_softmax(-1)


def images_to_tensor(images: Sequence[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad grayscale line images to a common width; returns ``(batch, widths)``.

    Pixels are scaled to [-1, 1]; padding is white like the crop padding.
    """
    widths = [im.width for im in images]
    height = images[0].height
    if any(im.height != height for im in images):
        raise ValueError("all images in a batch must have the same height")
    max_w = max(widths) + MIN_RIGHT_PAD
    max_w += -max_w % WIDTH_DOWNSAMPLE
    batch = np.full((len(images), 1, height, max_w), PAD_VALUE, dtype=np.uint8)
    for i, im in enumerate(images):
        batch[i, 0, :, : im.width] = np.asarray(im.convert("L"))
    tensor = torch.from_numpy(batch).float().div_(127.5).sub_(1.0)
    return tensor, torch.tensor(widths, dtype=torch.long)


def greedy_decode(
    log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary
) -> list[tuple[str, float]]:
    """Best-path CTC decoding of ``(T, B, C)`` log-probs.

    Confidence is the geometric mean of the per-frame max probabilities, and
    0.0 for an empty prediction (an all-blank path is not a confident read).
    """
    best, idx = log_probs.detach().max(-1)  # (T, B)
    out = []
    for b in range(idx.size(1)):
        n = int(lengths[b])
        text = vocab.ctc_decode(idx[:n, b].tolist())
        confidence = float(best[:n, b].mean().exp()) if text else 0.0
        out.append((text, confidence))
    return out


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: CRNN,
    vocab: Vocabulary,
    *,
    preprocessing: dict[str, int] | None = None,
    **extra: Any,
) -> Path:
    """Save weights plus everything needed to rebuild the recognizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.config)
    cfg["cnn_channels"] = list(cfg["cnn_channels"])
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": cfg,
            "alphabet": vocab.alphabet,
            "preprocessing": preprocessing
            or {
                "target_height": DEFAULT_TARGET_HEIGHT,
                "horizontal_padding": DEFAULT_HORIZONTAL_PADDING,
                "vertical_padding": DEFAULT_VERTICAL_PADDING,
            },
            **extra,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[CRNN, Vocabulary]:
    cfg = dict(checkpoint["model_config"])
    cfg["cnn_channels"] = tuple(cfg["cnn_channels"])
    model = CRNN(CRNNConfig(**cfg))
    model.load_state_dict(checkpoint["model_state"])
    return model, Vocabulary(checkpoint["alphabet"])


# ---------------------------------------------------------------------------
# TextRecognizer adapter
# ---------------------------------------------------------------------------


class CRNNRecognizer(TextRecognizer):
    """Recognize raw field crops with a trained CRNN checkpoint.

    Crops (uint8 grayscale or RGB arrays) go through the same
    ``preprocess_ocr_crop`` used to build the training data.
    """

    def __init__(
        self,
        model: CRNN,
        vocab: Vocabulary,
        *,
        preprocessing: dict[str, int] | None = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.preprocessing = preprocessing or {}
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu", **kwargs: Any) -> "CRNNRecognizer":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Recognizer checkpoint not found: {path}")
        checkpoint = load_checkpoint(path)
        model, vocab = model_from_checkpoint(checkpoint)
        return cls(model, vocab, preprocessing=checkpoint.get("preprocessing"), device=device, **kwargs)

    def _prepare(self, crop: ImageArray) -> Image.Image:
        if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8:
            raise TypeError("crop must be a uint8 numpy array")
        if crop.ndim not in (2, 3) or crop.size == 0:
            raise ValueError(f"crop must be a non-empty HxW or HxWxC array, got shape {crop.shape}")
        return preprocess_ocr_crop(Image.fromarray(crop), **self.preprocessing)

    def recognize(self, crop: ImageArray) -> RecognitionResult:
        return self.recognize_batch([crop])[0]

    @torch.inference_mode()
    def recognize_batch(self, crops: Sequence[ImageArray]) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for start in range(0, len(crops), self.batch_size):
            images = [self._prepare(c) for c in crops[start : start + self.batch_size]]
            batch, widths = images_to_tensor(images)
            log_probs = self.model(batch.to(self.device), widths.to(self.device))
            for text, confidence in greedy_decode(log_probs, CRNN.output_lengths(widths), self.vocab):
                results.append(RecognitionResult(text=text, confidence=confidence))
        return results
