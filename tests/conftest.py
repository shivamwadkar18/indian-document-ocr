from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_image():
    """Write a small solid-colour image to ``path`` (test fixture, not dataset content)."""

    def _make(path: Path, size: tuple[int, int] = (32, 16), fmt: str | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((size[1], size[0], 3), 128, dtype=np.uint8)).save(path, format=fmt)
        return path

    return _make
