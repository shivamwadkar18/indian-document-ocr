"""End-to-end document pipeline.

    image
      -> preprocessing
      -> text (field) detection
      -> document classification   (skipped if document_type is given; inferred
                                     from detected fields if no classifier)
      -> crop (clipped to the image) + text recognition
      -> document-specific extraction
      -> validation
      -> PipelineResult  (.to_dict() -> JSON-ready)

Every stage is an injected component implementing an interface, so any one can
be swapped without touching the others. This class is the intended boundary
for a future backend: ``DocumentPipeline.run(image).to_dict()``.

Problems that do not stop the run (detections that cannot be cropped or have
an unknown class, missing or unexpected fields, an ambiguous document type)
are reported in ``PipelineResult.issues``. Missing fields stay ``None``.

``from_assets`` / ``from_config`` load the trained YOLO detector and CRNN
recognizer. Their frameworks are imported lazily so this module stays
importable without PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import UnidentifiedImageError

from idocr.data.annotation_schema import DETECTOR_CLASSES, DOCUMENT_FIELDS, get_valid_fields
from idocr.data.preprocessing import Preprocessor
from idocr.extraction import AadhaarExtractor, PanExtractor
from idocr.extraction.base import FieldExtractor
from idocr.models import DocumentClassifier, TextDetector, TextRecognizer
from idocr.types import (
    ClassificationResult,
    DocumentType,
    ImageArray,
    PipelineResult,
    RecognizedRegion,
    TextRegion,
    ValidationIssue,
)
from idocr.utils.config import load_config
from idocr.utils.image import crop, image_size, load_image
from idocr.validation import Validator

#: Crops narrower or shorter than this (after clipping) are not recognized.
MIN_CROP_SIDE = 2

#: Document types whose fields come from the field-level detector.
SUPPORTED_DOCUMENTS = (DocumentType.AADHAAR, DocumentType.PAN)


class PipelineConfigurationError(RuntimeError):
    pass


def check_image(image: Any) -> None:
    """Raise if ``image`` is not a non-empty H x W x 3 uint8 array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape H x W x 3, got {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got {image.shape}")


def _exclusive_fields() -> dict[DocumentType, set[str]]:
    """Fields that identify one supported document type on their own."""
    out = {}
    for doc in SUPPORTED_DOCUMENTS:
        others = set().union(*(DOCUMENT_FIELDS[d] for d in SUPPORTED_DOCUMENTS if d is not doc))
        out[doc] = DOCUMENT_FIELDS[doc] - others
    return out


def infer_document_type(regions: list[TextRegion]) -> tuple[ClassificationResult, list[ValidationIssue]]:
    """Aadhaar vs PAN from detected field classes (no classifier model yet).

    Each type scores the highest detection confidence among the fields only
    it has (Aadhaar: aadhaar_number, gender; PAN: pan_number, fathers_name).
    """
    scores = {
        doc.value: max((r.confidence for r in regions if r.class_name in fields), default=0.0)
        for doc, fields in _exclusive_fields().items()
    }
    found = {k: v for k, v in scores.items() if v > 0}
    if not found:
        return (
            ClassificationResult(DocumentType.UNKNOWN, 0.0, scores),
            [ValidationIssue("unknown_document_type",
                             "no Aadhaar- or PAN-specific field detected; document type unknown")],
        )
    best = max(found, key=found.get)
    issues = []
    if len(found) > 1:
        issues.append(ValidationIssue("ambiguous_document_type",
                                      f"fields of both document types detected (scores {scores}); "
                                      f"using {best}"))
    return ClassificationResult(DocumentType(best), found[best], scores), issues


class DocumentPipeline:
    def __init__(
        self,
        *,
        detector: TextDetector,
        recognizer: TextRecognizer,
        classifier: DocumentClassifier | None = None,
        preprocessor: Preprocessor | None = None,
        extractors: Mapping[DocumentType, FieldExtractor] | None = None,
        validators: Mapping[DocumentType, Validator] | None = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.extractors = dict(extractors or {})
        self.validators = dict(validators or {})

    # ------------------------------------------------------------------ run

    def run(self, image: ImageArray, document_type: DocumentType | str | None = None) -> PipelineResult:
        """Process one RGB image.

        ``document_type``: pass it to skip classification. Otherwise the
        classifier is used, or, without one, the type is inferred from the
        detected fields.
        """
        check_image(image)
        if self.preprocessor is not None:
            image = self.preprocessor(image)
            check_image(image)
        size = image_size(image)
        issues: list[ValidationIssue] = []

        detected = self.detector.detect(image)

        classification = None
        if document_type is not None:
            document_type = DocumentType(document_type)
        elif self.classifier is not None:
            classification = self.classifier.classify(image)
            document_type = classification.document_type
        else:
            classification, type_issues = infer_document_type(detected)
            document_type = classification.document_type
            issues += type_issues

        regions = self._recognize(image, detected, issues)
        result = PipelineResult(
            document_type=document_type,
            image_size=size,
            classification=classification,
            regions=regions,
            issues=issues,
        )

        # Unknown documents or types without an extractor return OCR output only.
        extractor = self.extractors.get(document_type)
        if extractor is not None:
            result.fields = extractor.extract(regions, size)
            self._field_issues(result)
            validator = self.validators.get(document_type)
            if validator is not None:
                result.validation = validator.validate(result.fields)
        return result

    def run_file(self, path: str | Path, document_type: DocumentType | str | None = None) -> PipelineResult:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        try:
            image = load_image(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"cannot read image {path}: {exc}") from exc
        return self.run(image, document_type)

    def _recognize(
        self, image: ImageArray, detected: list[TextRegion], issues: list[ValidationIssue]
    ) -> list[RecognizedRegion]:
        """Crop and recognize usable detections; report the rest as issues."""
        usable: list[TextRegion] = []
        crops: list[ImageArray] = []
        for region in detected:
            box = [round(v, 1) for v in region.bbox.to_list()]
            # Class-agnostic detectors (class_id -1, no name) are allowed; a
            # class the mapping does not know, or a mismatched name, is not.
            if region.class_id != -1 or region.class_name is not None:
                if DETECTOR_CLASSES.get(region.class_id) != region.class_name:
                    issues.append(ValidationIssue(
                        "unknown_class",
                        f"detection with class {region.class_id} ({region.class_name!r}) at {box} "
                        f"does not match the detector class mapping; skipped",
                        region.class_name))
                    continue
            patch = crop(image, region.bbox)
            if patch.shape[0] < MIN_CROP_SIDE or patch.shape[1] < MIN_CROP_SIDE:
                issues.append(ValidationIssue(
                    "empty_crop",
                    f"detection at {box} is empty or outside the image after clipping; skipped",
                    region.class_name))
                continue
            usable.append(region)
            crops.append(np.ascontiguousarray(patch))

        recognitions = self.recognizer.recognize_batch(crops) if crops else []
        return [RecognizedRegion(r, t) for r, t in zip(usable, recognitions, strict=True)]

    @staticmethod
    def _field_issues(result: PipelineResult) -> None:
        expected = get_valid_fields(result.document_type)
        for name, field in result.fields.items():
            if field.value is None:
                result.issues.append(ValidationIssue(
                    "missing_field", f"{name} was not detected or not readable", name))
        counts: dict[str, int] = {}
        for region in result.regions:
            name = region.region.class_name
            if name is not None:
                counts[name] = counts.get(name, 0) + 1
        for name, count in counts.items():
            if name not in expected:
                result.issues.append(ValidationIssue(
                    "unexpected_field",
                    f"{name} detected but not a {result.document_type.value} field; ignored", name))
            elif count > 1:
                result.issues.append(ValidationIssue(
                    "duplicate_field",
                    f"{count} {name} detections; used the most confident", name))

    # -------------------------------------------------------------- loading

    @classmethod
    def from_assets(
        cls,
        detector_weights: str | Path,
        recognizer_checkpoint: str | Path,
        *,
        device: str = "cpu",
        detector_options: Mapping[str, Any] | None = None,
        recognizer_options: Mapping[str, Any] | None = None,
    ) -> "DocumentPipeline":
        """Load the trained YOLO field detector and CRNN recognizer.

        ``detector_options`` go to ``UltralyticsFieldDetector`` (imgsz, conf,
        iou, max_det); ``recognizer_options`` to ``CRNNRecognizer`` (batch_size).

        Raises:
            FileNotFoundError: if either asset is missing (both are checked
                before anything is loaded).
            PipelineConfigurationError: if the detector's class names differ
                from the project class mapping.
        """
        missing = [f"{label}: {p}" for label, p in
                   (("detector weights", Path(detector_weights)),
                    ("recognizer checkpoint", Path(recognizer_checkpoint)))
                   if not p.is_file()]
        if missing:
            raise FileNotFoundError("model asset(s) not found - " + "; ".join(missing))

        from idocr.models.detector.yolo import UltralyticsFieldDetector
        from idocr.models.recognizer.crnn import CRNNRecognizer

        detector = UltralyticsFieldDetector(detector_weights, device=device, **dict(detector_options or {}))
        if detector.class_names != DETECTOR_CLASSES:
            raise PipelineConfigurationError(
                f"detector classes {detector.class_names} do not match the project mapping {DETECTOR_CLASSES}")
        recognizer = CRNNRecognizer.from_checkpoint(recognizer_checkpoint, device=device,
                                                    **dict(recognizer_options or {}))
        return cls(
            detector=detector,
            recognizer=recognizer,
            extractors={DocumentType.AADHAAR: AadhaarExtractor(), DocumentType.PAN: PanExtractor()},
        )

    @classmethod
    def from_config(cls, path: str | Path, root: str | Path | None = None,
                    **overrides: Any) -> "DocumentPipeline":
        """``from_assets`` with settings from a YAML file (see ``configs/inference.yaml``).

        Relative asset paths resolve against ``root`` (default: the project
        root). Non-None ``overrides`` (detector_weights, recognizer_checkpoint,
        device) win over the file.
        """
        from idocr.utils.paths import find_project_root

        cfg = {**load_config(path), **{k: v for k, v in overrides.items() if v is not None}}
        unknown = sorted(set(cfg) - {"detector_weights", "recognizer_checkpoint", "device",
                                     "detector", "recognizer"})
        if unknown:
            raise PipelineConfigurationError(f"unknown inference config keys: {unknown}")
        for key in ("detector_weights", "recognizer_checkpoint"):
            if not cfg.get(key):
                raise PipelineConfigurationError(f"{key} is required")
        base = Path(root) if root is not None else find_project_root()

        def resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else base / p

        return cls.from_assets(
            resolve(cfg["detector_weights"]),
            resolve(cfg["recognizer_checkpoint"]),
            device=cfg.get("device", "cpu"),
            detector_options=cfg.get("detector"),
            recognizer_options=cfg.get("recognizer"),
        )
