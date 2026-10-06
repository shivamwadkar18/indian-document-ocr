"""Project root discovery and standard project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Environment variable that overrides project-root discovery.
ROOT_ENV_VAR = "IDOCR_PROJECT_ROOT"


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "idocr").is_dir()


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project root (directory containing pyproject.toml and src/idocr).

    Resolution order: ``IDOCR_PROJECT_ROOT`` env var, then walking up from
    ``start`` (if given), this file's location, and the current directory.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        root = Path(env).resolve()
        if not _is_project_root(root):
            raise FileNotFoundError(f"{ROOT_ENV_VAR}={env} is not an indian-document-ocr project root")
        return root

    starts = [Path(start)] if start is not None else [Path(__file__), Path.cwd()]
    for s in starts:
        s = s.resolve()
        for candidate in (s, *s.parents):
            if _is_project_root(candidate):
                return candidate
    raise FileNotFoundError("Could not locate project root (pyproject.toml + src/idocr)")


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths to the standard project directories."""

    root: Path
    raw: Path
    processed: Path
    synthetic: Path
    reports: Path
    models: Path
    experiments: Path
    configs: Path

    @classmethod
    def from_config(cls, config: Mapping[str, Any], root: str | Path | None = None) -> "ProjectPaths":
        root_path = Path(root).resolve() if root is not None else find_project_root()
        p = config.get("paths", {})

        def resolve(key: str, default: str) -> Path:
            value = Path(p.get(key, default))
            return value if value.is_absolute() else (root_path / value).resolve()

        return cls(
            root=root_path,
            raw=resolve("raw", "data/raw"),
            processed=resolve("processed", "data/processed"),
            synthetic=resolve("synthetic", "data/synthetic"),
            reports=resolve("reports", "data/reports"),
            models=resolve("models", "models"),
            experiments=resolve("experiments", "experiments"),
            configs=root_path / "configs",
        )

    def raw_dir(self, document_type: str) -> Path:
        """Raw dataset directory for one document type, e.g. data/raw/aadhaar."""
        return self.raw / document_type
