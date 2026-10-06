"""Audit report container and JSON writer.

Human-readable Markdown is produced across datasets by ``audit.summary``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPORT_SUFFIX = "_report.json"


@dataclass
class AuditReport:
    dataset_name: str
    dataset_root: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sections: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_section(self, name: str, content: Any) -> None:
        self.sections[name] = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_root": self.dataset_root,
            "created_at": self.created_at,
            "notes": self.notes,
            "sections": self.sections,
        }

    def write(self, reports_dir: str | Path) -> Path:
        """Write (overwrite) ``<reports_dir>/<dataset_name>_report.json``."""
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{self.dataset_name}{REPORT_SUFFIX}"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path
