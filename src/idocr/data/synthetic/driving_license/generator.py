"""Synthetic Driving Licence generator orchestration (not implemented)."""

from __future__ import annotations

from idocr.data.synthetic.driving_license.config import GeneratorConfig


class SyntheticDrivingLicenseGenerator:
    """identity -> layout render -> degradation -> (image, annotation).

    TODO:
      1. IdentityGenerator implementation (fictional data only)
      2. Layout templates + renderer (fonts, sizes, spacing/position jitter)
      3. Degradation pipeline (see augmentations.DegradationConfig)
      4. Emit annotations in the unified schema (after it is finalised)
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def generate(self) -> None:
        raise NotImplementedError(
            "Synthetic DL generation is not implemented yet; see TODOs in this module."
        )
