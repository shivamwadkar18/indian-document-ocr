"""List which raw datasets are present under data/raw/ (no file contents are read).

Usage:
    python scripts/discover_datasets.py
"""

from __future__ import annotations

from idocr.data.discovery import discover_datasets
from idocr.utils import ProjectPaths, load_default_config


def main() -> None:
    config = load_default_config()
    paths = ProjectPaths.from_config(config)
    for loc in discover_datasets(paths.raw, config["documents"]):
        status = "missing" if not loc.exists else ("empty" if loc.is_empty else f"{loc.file_count} files")
        print(f"{loc.document_type:<16} {status:<12} {loc.path}")


if __name__ == "__main__":
    main()
