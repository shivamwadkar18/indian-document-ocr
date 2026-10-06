"""Shared utilities: project paths, configuration, image I/O, logging."""

from idocr.utils.config import deep_merge, load_config, load_default_config
from idocr.utils.paths import ProjectPaths, find_project_root

__all__ = [
    "ProjectPaths",
    "deep_merge",
    "find_project_root",
    "load_config",
    "load_default_config",
]
