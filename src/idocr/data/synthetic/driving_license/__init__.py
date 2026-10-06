"""Synthetic Driving Licence generator (scaffold — nothing is generated yet).

Components
----------
config        GeneratorConfig (loaded from configs/synthetic_driving_license.yaml)
identity      fictional identity / licence data
layouts       document layout templates (field positions, fonts)
augmentations photo/scan degradation settings
generator     orchestration: identity -> layout render -> degrade -> image + annotation

Rules: fictional data only; never reproduce real people's credentials; never
use real licence scans as templates.
"""

from idocr.data.synthetic.driving_license.config import GeneratorConfig
from idocr.data.synthetic.driving_license.generator import SyntheticDrivingLicenseGenerator
from idocr.data.synthetic.driving_license.identity import FictionalIdentity, IdentityGenerator

__all__ = ["FictionalIdentity", "GeneratorConfig", "IdentityGenerator", "SyntheticDrivingLicenseGenerator"]
