"""Directory structure summary of a dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from idocr.data.discovery import iter_files


@dataclass
class TreeSummary:
    root: Path
    total_files: int
    extension_counts: dict[str, int]
    # relative directory -> {extension: count}
    directories: dict[str, dict[str, int]] = field(default_factory=dict)
    # directory names that look like split names (reporting only; not assumed)
    candidate_splits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "total_files": self.total_files,
            "extension_counts": self.extension_counts,
            "directories": self.directories,
            "candidate_splits": self.candidate_splits,
        }


def extension_of(path: Path) -> str:
    return path.suffix.lower() or "<none>"


def scan_tree(root: str | Path, split_name_candidates: Iterable[str] = ()) -> TreeSummary:
    root = Path(root)
    split_names = {s.lower() for s in split_name_candidates}
    ext_counts: Counter[str] = Counter()
    per_dir: dict[str, Counter[str]] = {}
    splits: set[str] = set()

    for path in iter_files(root):
        rel_dir = path.parent.relative_to(root).as_posix() or "."
        ext = extension_of(path)
        ext_counts[ext] += 1
        per_dir.setdefault(rel_dir, Counter())[ext] += 1
        for part in path.parent.relative_to(root).parts:
            if part.lower() in split_names:
                splits.add(part)

    return TreeSummary(
        root=root,
        total_files=sum(ext_counts.values()),
        extension_counts=dict(sorted(ext_counts.items())),
        directories={d: dict(sorted(c.items())) for d, c in sorted(per_dir.items())},
        candidate_splits=sorted(splits),
    )
