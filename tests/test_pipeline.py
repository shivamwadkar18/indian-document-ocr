"""Pipeline wiring tests using trivial stub components (no models, no data)."""

import json

import numpy as np
import pytest

from idocr.extraction import default_extractors
from idocr.extraction.base import FieldExtractor
from idocr.models import DocumentClassifier, TextDetector, TextRecognizer
from idocr.pipeline import DocumentPipeline, PipelineConfigurationError
from idocr.types import (
    BBox,
    ClassificationResult,
    DocumentType,
    ExtractedField,
    RecognitionResult,
    TextRegion,
)


class StubClassifier(DocumentClassifier):
    def classify(self, image):
        return ClassificationResult(DocumentType.PAN, 0.9)


class StubDetector(TextDetector):
    def detect(self, image):
        return [TextRegion(BBox(0, 0, 10, 5), 0.8), TextRegion(BBox(5, 5, 20, 10), 0.7)]


class ShapeRecognizer(TextRecognizer):
    """Returns the crop shape as text, to prove crops are taken from the image."""

    def recognize(self, crop):
        return RecognitionResult(f"{crop.shape[1]}x{crop.shape[0]}", 0.5)


class CountExtractor(FieldExtractor):
    document_type = DocumentType.PAN

    def extract(self, regions, image_size):
        return {"n": ExtractedField("n", str(len(regions)), source_regions=list(range(len(regions))))}


@pytest.fixture
def image():
    return np.zeros((16, 32, 3), dtype=np.uint8)


def test_pipeline_end_to_end_with_stubs(image):
    pipe = DocumentPipeline(
        classifier=StubClassifier(),
        detector=StubDetector(),
        recognizer=ShapeRecognizer(),
        extractors={DocumentType.PAN: CountExtractor()},
    )
    result = pipe.run(image)
    assert result.document_type is DocumentType.PAN
    assert [r.recognition.text for r in result.regions] == ["10x5", "15x5"]
    assert result.fields["n"].value == "2"
    json.dumps(result.to_dict())  # JSON-serialisable


def test_pipeline_requires_classifier_or_type(image):
    pipe = DocumentPipeline(detector=StubDetector(), recognizer=ShapeRecognizer())
    with pytest.raises(PipelineConfigurationError):
        pipe.run(image)
    result = pipe.run(image, document_type=DocumentType.UNKNOWN)
    assert result.fields == {} and len(result.regions) == 2


def test_default_extractors_are_unimplemented_placeholders():
    extractors = default_extractors()
    assert set(extractors) == {DocumentType.AADHAAR, DocumentType.PAN, DocumentType.DRIVING_LICENSE}
    for ex in extractors.values():
        with pytest.raises(NotImplementedError):
            ex.extract([], (1, 1))
