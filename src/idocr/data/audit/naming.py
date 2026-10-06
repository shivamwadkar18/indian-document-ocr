"""Filename conventions observed in the real datasets.

Roboflow exports name files ``<original name with '.' -> '_'>.rf.<32 hex>``.
Several exported files can derive from the same original upload (augmented
variants), so the original-name part is used to detect cross-split leakage.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_ROBOFLOW_STEM = re.compile(r"^(?P<source>.+)\.rf\.(?P<hash>[0-9a-f]{32})$")


def roboflow_source_stem(stem: str) -> str | None:
    """Original-upload part of a Roboflow export stem, or None if not Roboflow-named."""
    m = _ROBOFLOW_STEM.match(stem)
    return m.group("source") if m else None


def case_insensitive_collisions(paths: Iterable[Path]) -> list[list[str]]:
    """Groups of filenames in the same directory that differ only by case."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in paths:
        groups[(p.parent.as_posix().lower(), p.name.lower())].append(p.name)
    return [sorted(v) for v in groups.values() if len(v) > 1]
