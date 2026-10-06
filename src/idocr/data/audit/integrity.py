"""Source-data integrity: hash every file before and after an audit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from idocr.data.discovery import iter_files


def hash_tree(root: Path) -> dict[str, tuple[str, int, int]]:
    """relative path -> (sha256, size, mtime_ns) for every data file under ``root``."""
    out = {}
    for p in iter_files(root):
        st = p.stat()
        out[p.relative_to(root).as_posix()] = (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns)
    return out


@dataclass
class IntegrityResult:
    files_checked: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def to_dict(self) -> dict:
        return {
            "files_checked": self.files_checked,
            "source_data_modified": self.modified,
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
        }


def compare_trees(before: dict, after: dict) -> IntegrityResult:
    return IntegrityResult(
        files_checked=len(before),
        added=sorted(set(after) - set(before)),
        removed=sorted(set(before) - set(after)),
        changed=sorted(k for k in set(before) & set(after) if before[k] != after[k]),
    )
