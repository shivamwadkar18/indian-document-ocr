"""End-to-end document pipeline.

    image
      -> preprocessing
      -> document classification   (skipped if document_type is given)
      -> text detection
      -> text recognition
      -> document-specific extraction
      -> validation
      -> PipelineResult  (.to_dict() -> JSON-ready)

Every stage is an injected component implementing an interface, so any one can
be swapped without touching the others. This class is the intended boundary
for a future backend: ``DocumentPipeline.run(image).to_dict()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from idocr.data.preprocessing import Preprocessor
from idocr.extraction.base import FieldExtractor
from idocr.models import DocumentClassifier, TextDetector, TextRecognizer
from idocr.types import DocumentType, ImageArray, PipelineResult, RecognizedRegion
from idocr.utils.image import crop, image_size, load_image
from idocr.validation import Validator


class PipelineConfigurationError(RuntimeError):
    pass


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

    def run(self, image: ImageArray, document_type: DocumentType | None = None) -> PipelineResult:
        """Process one RGB image. Pass ``document_type`` to skip classification."""
        if self.preprocessor is not None:
            image = self.preprocessor(image)
        size = image_size(image)

        classification = None
        if document_type is None:
            if self.classifier is None:
                raise PipelineConfigurationError("No classifier configured and no document_type given.")
            classification = self.classifier.classify(image)
            document_type = classification.document_type

        detected = self.detector.detect(image)
        recognitions = self.recognizer.recognize_batch([crop(image, r.bbox) for r in detected])
        regions = [RecognizedRegion(r, t) for r, t in zip(detected, recognitions, strict=True)]

        result = PipelineResult(
            document_type=document_type,
            image_size=size,
            classification=classification,
            regions=regions,
        )
        # Unknown documents or types without an extractor return OCR output only.
        extractor = self.extractors.get(document_type)
        if extractor is not None:
            result.fields = extractor.extract(regions, size)
            validator = self.validators.get(document_type)
            if validator is not None:
                result.validation = validator.validate(result.fields)
        return result

    def run_file(self, path: str | Path, document_type: DocumentType | None = None) -> PipelineResult:
        return self.run(load_image(path), document_type)
