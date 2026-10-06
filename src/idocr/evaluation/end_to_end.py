"""End-to-end (structured output) metrics.

Implemented: per-field accuracy and document-level accuracy over exact string
match of field values.
TODO(after audit): per-field normalisation/comparison rules (dates, IDs, names),
confidence calibration and error analysis reports.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

FieldValues = Mapping[str, str | None]


def field_accuracy(predictions: Sequence[FieldValues], references: Sequence[FieldValues]) -> dict[str, float]:
    """Accuracy per field name, over documents where the reference has that field."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for pred, ref in zip(predictions, references):
        for name, value in ref.items():
            total[name] += 1
            correct[name] += int(pred.get(name) == value)
    return {name: correct[name] / total[name] for name in sorted(total)}


def document_accuracy(predictions: Sequence[FieldValues], references: Sequence[FieldValues]) -> float:
    """Fraction of documents where every reference field is predicted exactly."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not references:
        return 0.0
    ok = sum(all(p.get(k) == v for k, v in r.items()) for p, r in zip(predictions, references))
    return ok / len(references)


def confidence_error_analysis(*args, **kwargs) -> dict:
    raise NotImplementedError("Confidence/error analysis is defined once models produce outputs.")
