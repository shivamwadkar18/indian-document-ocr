"""Locate raw datasets on disk without interpreting their contents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

#: Files that exist only to keep directories in Git; never counted as data.
IGNORED_FILENAMES = frozenset({".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini"})


@dataclass(frozen=True)
class DatasetLocation:
    document_type: str
    path: Path
    exists: bool
    file_count: int

    @property
    def is_empty(self) -> bool:
        return self.file_count == 0


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every data file under ``root`` (recursive, sorted, placeholders skipped)."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in IGNORED_FILENAMES:
            yield path


def discover_datasets(raw_root: str | Path, document_types: Iterable[str]) -> list[DatasetLocation]:
    """Report, for each document type, whether ``raw_root/<type>`` exists and holds files."""
    raw_root = Path(raw_root)
    locations = []
    for doc_type in document_types:
        path = raw_root / doc_type
        exists = path.is_dir()
        count = sum(1 for _ in iter_files(path)) if exists else 0
        locations.append(DatasetLocation(doc_type, path, exists, count))
    return locations
