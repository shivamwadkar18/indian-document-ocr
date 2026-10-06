"""OCR metrics: CER, WER, exact match.

Metrics are corpus-level (total edits / total reference length).
TODO(after audit): text normalisation policy (case, whitespace, punctuation,
Unicode normalisation for Indic scripts) — currently none is applied.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def levenshtein(a: Sequence[T], b: Sequence[T]) -> int:
    """Edit distance (insert/delete/substitute, unit cost)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _error_rate(pairs: list[tuple[Sequence, Sequence]]) -> float:
    edits = sum(levenshtein(p, r) for p, r in pairs)
    total = sum(len(r) for _, r in pairs)
    if total == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / total


def _check(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    _check(predictions, references)
    return _error_rate(list(zip(predictions, references)))


def word_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    _check(predictions, references)
    return _error_rate([(p.split(), r.split()) for p, r in zip(predictions, references)])


def exact_match_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    _check(predictions, references)
    if not references:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)
