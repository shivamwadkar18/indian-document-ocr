"""OCR recognizer training (CRNN + CTC). See ``train.py`` for the pipeline."""

from idocr.training.recognition.config import (
    RecognizerConfigError,
    RecognizerTrainConfig,
    config_from_dict,
    load_recognizer_config,
)
from idocr.training.recognition.train import evaluate, recognition_metrics, train

__all__ = [
    "RecognizerConfigError",
    "RecognizerTrainConfig",
    "config_from_dict",
    "evaluate",
    "load_recognizer_config",
    "recognition_metrics",
    "train",
]
