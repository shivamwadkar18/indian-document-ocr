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


def test_pipeline_without_classifier_or_type_reports_unknown(image):
    # Class-agnostic detections carry no field identity, so the type stays unknown.
    pipe = DocumentPipeline(detector=StubDetector(), recognizer=ShapeRecognizer())
    result = pipe.run(image)
    assert result.document_type is DocumentType.UNKNOWN
    assert [i.code for i in result.issues] == ["unknown_document_type"]
    result = pipe.run(image, document_type=DocumentType.UNKNOWN)
    assert result.fields == {} and len(result.regions) == 2


def test_default_extractors():
    extractors = default_extractors()
    assert set(extractors) == {DocumentType.AADHAAR, DocumentType.PAN, DocumentType.DRIVING_LICENSE}
    assert set(extractors[DocumentType.AADHAAR].extract([], (1, 1))) == {
        "name", "date_of_birth", "gender", "aadhaar_number"}
    assert set(extractors[DocumentType.PAN].extract([], (1, 1))) == {
        "name", "date_of_birth", "pan_number", "fathers_name"}
    with pytest.raises(NotImplementedError):
        extractors[DocumentType.DRIVING_LICENSE].extract([], (1, 1))


# ---------------------------------------------------------------------------
# Field-level Aadhaar / PAN inference (mocked detector, no real data or weights)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402

from idocr.data.annotation_schema import DETECTOR_CLASSES  # noqa: E402
from idocr.extraction import AadhaarExtractor, PanExtractor  # noqa: E402
from idocr.pipeline.document_pipeline import check_image  # noqa: E402

FIELD_EXTRACTORS = {DocumentType.AADHAAR: AadhaarExtractor(), DocumentType.PAN: PanExtractor()}
CLASS_IDS = {name: cid for cid, name in DETECTOR_CLASSES.items()}


def field(class_name, box, conf=0.9, class_id=None):
    cid = class_id if class_id is not None else CLASS_IDS.get(class_name, 99)
    return TextRegion(BBox(*box), conf, class_id=cid, class_name=class_name)


class ListDetector(TextDetector):
    def __init__(self, regions):
        self.regions = regions

    def detect(self, image):
        return list(self.regions)


class RecordingRecognizer(TextRecognizer):
    """Reads each crop's mean grey level back as text and records the crops."""

    def __init__(self, texts=None):
        self.texts = texts
        self.crops = []

    def recognize(self, crop):
        raise AssertionError("pipeline must use recognize_batch")

    def recognize_batch(self, crops):
        self.crops.extend(crops)
        for c in crops:
            assert c.shape[0] >= 2 and c.shape[1] >= 2
        texts = self.texts or [f"T{int(c.mean())}" for c in crops]
        return [RecognitionResult(t, 0.5) for t in texts[: len(crops)]]


@pytest.fixture
def card():
    # Each field band has a distinct grey level so crops can be traced to boxes.
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    for i, level in enumerate((10, 20, 30, 40)):
        img[20 * i: 20 * i + 20, 0:100] = level
    return img


def pan_regions():
    return [
        field("name", (0, 0, 100, 20), 0.95),
        field("fathers_name", (0, 20, 100, 40), 0.9),
        field("date_of_birth", (0, 40, 100, 60), 0.85),
        field("pan_number", (0, 60, 100, 80), 0.8),
    ]


def pipeline(regions, recognizer=None):
    return DocumentPipeline(detector=ListDetector(regions), recognizer=recognizer or RecordingRecognizer(),
                            extractors=FIELD_EXTRACTORS)


def test_detector_to_recognizer_structured_output(card):
    recognizer = RecordingRecognizer()
    result = pipeline(pan_regions(), recognizer).run(card)

    assert result.document_type is DocumentType.PAN
    assert result.classification.scores["pan"] == pytest.approx(0.9)
    assert [c.shape for c in recognizer.crops] == [(20, 100, 3)] * 4
    assert {k: f.value for k, f in result.fields.items()} == {
        "name": "T10", "date_of_birth": "T30", "pan_number": "T40", "fathers_name": "T20"}
    assert result.fields["name"].confidence == pytest.approx(0.95 * 0.5)
    assert result.fields["pan_number"].source_regions == [3]
    assert result.issues == []

    out = result.to_dict()
    json.dumps(out)
    region = out["regions"][3]
    assert region["bbox"] == [0, 60, 100, 80]
    assert (region["class_id"], region["class_name"], region["text"]) == (4, "pan_number", "T40")
    assert region["detection_confidence"] == 0.8 and region["recognition_confidence"] == 0.5
    assert out["issues"] == []


def test_all_six_classes_keep_mapping(card):
    regions = [field(name, (0, 20 * (i % 4), 100, 20 * (i % 4) + 20)) for i, name in DETECTOR_CLASSES.items()]
    result = pipeline(regions).run(card, document_type="aadhaar")
    assert [(r.region.class_id, r.region.class_name) for r in result.regions] == list(DETECTOR_CLASSES.items())
    assert set(result.fields) == {"name", "date_of_birth", "gender", "aadhaar_number"}
    assert {i.field for i in result.issues if i.code == "unexpected_field"} == {"pan_number", "fathers_name"}


def test_aadhaar_inferred_and_missing_fields_not_fabricated(card):
    result = pipeline([field("aadhaar_number", (0, 0, 100, 20), 0.7), field("name", (0, 20, 100, 40))]).run(card)

    assert result.document_type is DocumentType.AADHAAR
    for name in ("date_of_birth", "gender"):
        f = result.fields[name]
        assert f.value is None and f.confidence is None and f.source_regions == []
    assert sorted(i.field for i in result.issues if i.code == "missing_field") == ["date_of_birth", "gender"]


def test_no_detections(card):
    recognizer = RecordingRecognizer()
    pipe = pipeline([], recognizer)

    result = pipe.run(card)
    assert result.document_type is DocumentType.UNKNOWN
    assert result.regions == [] and result.fields == {} and recognizer.crops == []

    result = pipe.run(card, document_type=DocumentType.PAN)
    assert all(f.value is None for f in result.fields.values())
    assert [i.code for i in result.issues].count("missing_field") == 4


def test_boxes_clipped_and_empty_crops_skipped(card):
    regions = [
        field("name", (-30, -10, 50, 20)),           # partly outside -> clipped
        field("pan_number", (150, 90, 400, 300)),    # partly outside -> clipped
        field("date_of_birth", (250, 10, 300, 30)),  # fully outside -> empty
        field("fathers_name", (10, 10, 11, 30)),     # 1 px wide -> too thin
        field("gender", (5, 5, 5, 5)),               # zero area
    ]
    recognizer = RecordingRecognizer()
    result = pipeline(regions, recognizer).run(card, document_type="pan")

    assert [c.shape[:2] for c in recognizer.crops] == [(20, 50), (10, 50)]
    assert [r.region.class_name for r in result.regions] == ["name", "pan_number"]
    empty = [i for i in result.issues if i.code == "empty_crop"]
    assert sorted(i.field for i in empty) == ["date_of_birth", "fathers_name", "gender"]
    assert result.fields["date_of_birth"].value is None
    assert result.fields["fathers_name"].value is None


def test_unknown_and_mismatched_classes_skipped(card):
    regions = [
        field("name", (0, 0, 100, 20)),
        field("signature", (0, 20, 100, 40), class_id=7),
        field("pan_number", (0, 40, 100, 60), class_id=0),  # id/name mismatch
    ]
    recognizer = RecordingRecognizer()
    result = pipeline(regions, recognizer).run(card, document_type="pan")
    assert len(recognizer.crops) == 1
    assert [i.code for i in result.issues].count("unknown_class") == 2


def test_duplicate_detections_use_most_confident(card):
    regions = [field("pan_number", (0, 0, 100, 20), 0.6), field("pan_number", (0, 20, 100, 40), 0.9)]
    result = pipeline(regions).run(card)
    assert result.fields["pan_number"].value == "T20"
    assert result.fields["pan_number"].source_regions == [1]
    assert any(i.code == "duplicate_field" for i in result.issues)


def test_empty_recognition_is_not_a_field_value(card):
    result = pipeline([field("pan_number", (0, 0, 100, 20))], RecordingRecognizer(texts=[""])).run(card)
    assert result.regions[0].recognition.text == ""
    assert result.fields["pan_number"].value is None
    assert any(i.code == "missing_field" and i.field == "pan_number" for i in result.issues)


def test_ambiguous_document_type_reported(card):
    regions = [field("pan_number", (0, 0, 100, 20), 0.6), field("aadhaar_number", (0, 20, 100, 40), 0.8)]
    result = pipeline(regions).run(card)
    assert result.document_type is DocumentType.AADHAAR
    assert any(i.code == "ambiguous_document_type" for i in result.issues)


@pytest.mark.parametrize("bad, error", [
    ("not an image", TypeError),
    (np.zeros((10, 10), dtype=np.uint8), ValueError),
    (np.zeros((10, 10, 4), dtype=np.uint8), ValueError),
    (np.zeros((10, 10, 3), dtype=np.float32), ValueError),
    (np.zeros((0, 10, 3), dtype=np.uint8), ValueError),
])
def test_invalid_input_images(bad, error):
    with pytest.raises(error):
        check_image(bad)
    with pytest.raises(error):
        pipeline([]).run(bad)


def test_invalid_document_type(card):
    with pytest.raises(ValueError):
        pipeline([]).run(card, document_type="passport")


def test_run_file_errors(tmp_path, card):
    pipe = pipeline(pan_regions())
    with pytest.raises(FileNotFoundError):
        pipe.run_file(tmp_path / "missing.jpg")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot read image"):
        pipe.run_file(broken)
    good = tmp_path / "card.png"
    Image.fromarray(card).save(good)
    assert pipe.run_file(good).fields["pan_number"].value == "T40"


# --- loading assets ----------------------------------------------------------


def test_from_assets_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        DocumentPipeline.from_assets(tmp_path / "det.pt", tmp_path / "rec.pt")
    assert "detector weights" in str(exc.value) and "recognizer checkpoint" in str(exc.value)


class FakeYolo(TextDetector):
    """Stands in for UltralyticsFieldDetector (no ultralytics, no weights)."""

    class_names = dict(DETECTOR_CLASSES)

    def __init__(self, model_path, *, device="cpu", **options):
        self.model_path, self.device, self.options = Path(model_path), device, options

    def detect(self, image):
        return pan_regions()


def _crnn_checkpoint(path: Path) -> Path:
    """A real (untrained, tiny) CRNN checkpoint in the production format."""
    pytest.importorskip("torch")
    from idocr.models.recognizer import Vocabulary
    from idocr.models.recognizer.crnn import CRNN, CRNNConfig, save_checkpoint

    vocab = Vocabulary()
    model = CRNN(CRNNConfig(num_classes=vocab.num_classes, hidden_size=16, lstm_layers=1))
    return save_checkpoint(path, model, vocab)


def test_from_assets_with_real_crnn_checkpoint(tmp_path, monkeypatch, card):
    import idocr.models.detector.yolo as yolo

    monkeypatch.setattr(yolo, "UltralyticsFieldDetector", FakeYolo)
    weights = tmp_path / "det.pt"
    weights.write_bytes(b"fake")

    pipe = DocumentPipeline.from_assets(weights, _crnn_checkpoint(tmp_path / "rec.pt"),
                                        detector_options={"conf": 0.3})
    assert pipe.detector.options == {"conf": 0.3}
    result = pipe.run(card)

    # Untrained CRNN: whatever it reads, the output is well-formed and nothing is invented.
    assert result.document_type is DocumentType.PAN
    assert len(result.regions) == 4
    for f in result.fields.values():
        assert (f.value is None) == (f.confidence is None)
        if f.value is not None:
            assert f.value == f.value.strip() and 0 <= f.confidence <= 1
    json.dumps(result.to_dict())


def test_from_assets_rejects_wrong_class_mapping(tmp_path, monkeypatch):
    import idocr.models.detector.yolo as yolo

    class WrongYolo(FakeYolo):
        class_names = {0: "name", 1: "pan_number"}

    monkeypatch.setattr(yolo, "UltralyticsFieldDetector", WrongYolo)
    for name in ("det.pt", "rec.pt"):
        (tmp_path / name).write_bytes(b"fake")
    with pytest.raises(PipelineConfigurationError, match="class"):
        DocumentPipeline.from_assets(tmp_path / "det.pt", tmp_path / "rec.pt")


def test_from_config(tmp_path, monkeypatch):
    import idocr.models.detector.yolo as yolo

    monkeypatch.setattr(yolo, "UltralyticsFieldDetector", FakeYolo)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "det.pt").write_bytes(b"fake")
    _crnn_checkpoint(tmp_path / "models" / "rec.pt")
    config = tmp_path / "inference.yaml"
    config.write_text("detector_weights: models/det.pt\nrecognizer_checkpoint: models/missing.pt\n"
                      "detector: {imgsz: 320}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        DocumentPipeline.from_config(config, root=tmp_path)
    pipe = DocumentPipeline.from_config(config, root=tmp_path, recognizer_checkpoint="models/rec.pt")
    assert pipe.detector.options == {"imgsz": 320}

    config.write_text("detector_weights: a.pt\nrecognizer_checkpoint: b.pt\ntypo: 1\n", encoding="utf-8")
    with pytest.raises(PipelineConfigurationError, match="typo"):
        DocumentPipeline.from_config(config, root=tmp_path)


def test_shipped_inference_config_loads_until_missing_weights(monkeypatch, tmp_path):
    import idocr.models.detector.yolo as yolo
    from idocr.utils.paths import find_project_root

    monkeypatch.setattr(yolo, "UltralyticsFieldDetector", FakeYolo)
    # The real weights live on Google Drive; with an empty root only the files are missing.
    with pytest.raises(FileNotFoundError, match="model asset"):
        DocumentPipeline.from_config(find_project_root() / "configs" / "inference.yaml", root=tmp_path)
