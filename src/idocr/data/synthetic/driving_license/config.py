"""Generator configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from idocr.data.synthetic.driving_license.augmentations import DegradationConfig


@dataclass
class GeneratorConfig:
    output_dir: Path = Path("data/synthetic/driving_license")
    num_samples: int = 0
    seed: int = 42
    licence_number_prefix: str = "XX"
    vehicle_categories: list[str] = field(default_factory=lambda: ["MCWG", "LMV", "MCWOG", "TRANS"])
    layouts: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    font_size_range: tuple[int, int] = (14, 28)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GeneratorConfig":
        identity = d.get("identity", {})
        return cls(
            output_dir=Path(d.get("output_dir", cls.output_dir)),
            num_samples=int(d.get("num_samples", 0)),
            seed=int(d.get("seed", 42)),
            licence_number_prefix=identity.get("licence_number_prefix", "XX"),
            vehicle_categories=list(identity.get("vehicle_categories", ["MCWG", "LMV", "MCWOG", "TRANS"])),
            layouts=list(d.get("layouts") or []),
            fonts=list(d.get("fonts") or []),
            font_size_range=tuple(d.get("font_size_range", (14, 28))),
            degradation=DegradationConfig.from_dict(d.get("augmentation", {})),
        )
