"""Photo / scan degradation settings for synthetic licences.

TODO: implement each degradation as an ``idocr.data.augmentation.Augmentation``.
Geometric ones (rotation, perspective) must also transform the generated
annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class DegradationConfig:
    rotation_deg: tuple[float, float] = (-5.0, 5.0)
    perspective: float = 0.0
    blur: float = 0.0
    noise: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    jpeg_quality: tuple[int, int] = (60, 95)
    shadow: float = 0.0
    scan_artifacts: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DegradationConfig":
        known = cls.__dataclass_fields__
        kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known}
        return cls(**kwargs)
