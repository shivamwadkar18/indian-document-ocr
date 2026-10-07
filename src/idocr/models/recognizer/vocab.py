"""OCR character vocabulary with a CTC blank (torch-free).

Index 0 is the CTC blank; characters occupy 1..len(alphabet).
"""

from __future__ import annotations

from typing import Iterable, Sequence

DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
BLANK_INDEX = 0


class Vocabulary:
    def __init__(self, alphabet: str = DEFAULT_ALPHABET) -> None:
        if not alphabet:
            raise ValueError("alphabet must not be empty")
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must not contain duplicate characters")
        self.alphabet = alphabet
        self._char_to_index = {c: i + 1 for i, c in enumerate(alphabet)}

    @property
    def num_classes(self) -> int:
        """Output classes including the blank."""
        return len(self.alphabet) + 1

    def __len__(self) -> int:
        return self.num_classes

    def encode(self, text: str) -> list[int]:
        """Map ``text`` to label indices.

        Raises:
            ValueError: for empty text or characters outside the alphabet.
        """
        if not text:
            raise ValueError("text must not be empty")
        unknown = sorted(set(text) - self._char_to_index.keys())
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        return [self._char_to_index[c] for c in text]

    def decode(self, indices: Iterable[int]) -> str:
        """Map label indices (no blanks, no CTC collapsing) back to text."""
        return "".join(self.alphabet[i - 1] for i in indices if i != BLANK_INDEX)

    def ctc_collapse(self, indices: Sequence[int]) -> list[int]:
        """Greedy CTC post-processing: merge repeats, then drop blanks."""
        out: list[int] = []
        prev = None
        for i in indices:
            if i != prev and i != BLANK_INDEX:
                out.append(i)
            prev = i
        return out

    def ctc_decode(self, indices: Sequence[int]) -> str:
        """Decode a per-frame best-path index sequence to text."""
        return self.decode(self.ctc_collapse(indices))
