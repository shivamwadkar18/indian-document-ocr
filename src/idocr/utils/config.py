"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from idocr.utils.paths import find_project_root

DEFAULT_CONFIG_NAME = "default.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file whose top level must be a mapping (empty file -> {})."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


def load_default_config(root: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/default.yaml`` from the project root."""
    root_path = Path(root) if root is not None else find_project_root()
    return load_config(root_path / "configs" / DEFAULT_CONFIG_NAME)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
