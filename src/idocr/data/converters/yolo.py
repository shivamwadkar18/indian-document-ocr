"""Converter from YOLO annotation format into Unified Annotation Schema."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image

from idocr.data.annotation_schema import (
    AnnotatedDocument,
    AnnotatedField,
    DocumentProvenance,
    FieldGeometry,
    FieldTranscription,
    TranscriptionStatus,
    is_valid_field,
)
from idocr.data.converters.base import DatasetConverter
from idocr.types import DocumentType


# Canonical class mappings from raw YOLO class IDs to semantic field names.
# None indicates the class is intentionally excluded from the unified field schema.
DEFAULT_AADHAAR_MAPPING: dict[int, str | None] = {
    0: "aadhaar_number",
    1: "date_of_birth",
    2: "gender",
    3: "name",
    4: None,  # Excluded unresolved annotation noise
}

DEFAULT_PAN_MAPPING: dict[int, str | None] = {
    0: "date_of_birth",
    1: "fathers_name",
    2: "name",
    3: "pan_number",
}


@dataclass
class ConversionStats:
    """Statistics recorded during dataset conversion."""
    document_type: str
    input_images: int = 0
    input_annotations: int = 0
    converted_documents: int = 0
    included_fields: int = 0
    excluded_class_4_annotations: int = 0
    conversion_errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.conversion_errors) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "input_images": self.input_images,
            "input_annotations": self.input_annotations,
            "converted_documents": self.converted_documents,
            "included_fields": self.included_fields,
            "excluded_class_4_annotations": self.excluded_class_4_annotations,
            "conversion_errors": list(self.conversion_errors),
        }


class YoloDatasetConverter(DatasetConverter):
    """Converts YOLO txt annotations and images into Unified Annotation Schema."""

    source_format: str = "yolo_txt"

    def __init__(
        self,
        document_type: DocumentType | str,
        class_mapping: Mapping[int, str | None] | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        if isinstance(document_type, str):
            self.document_type = DocumentType(document_type.lower())
        else:
            self.document_type = document_type

        if class_mapping is not None:
            self.class_mapping = dict(class_mapping)
        elif self.document_type == DocumentType.AADHAAR:
            self.class_mapping = DEFAULT_AADHAAR_MAPPING
        elif self.document_type == DocumentType.PAN:
            self.class_mapping = DEFAULT_PAN_MAPPING
        else:
            self.class_mapping = {}

        # Optional manifest for rich provenance lookup
        self._manifest_lookup: dict[str, dict[str, str]] = {}
        if manifest_path:
            self._load_manifest(Path(manifest_path))

    def _load_manifest(self, manifest_path: Path) -> None:
        import csv
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stem = Path(row.get("image_rel_path", "")).stem or row.get("filename", "")
                    if stem:
                        self._manifest_lookup[stem] = row

    def convert_file(
        self,
        image_path: Path,
        label_path: Path | None = None,
        source_split: str | None = None,
    ) -> tuple[AnnotatedDocument, int, int]:
        """Convert a single image and label file into an AnnotatedDocument.

        Returns:
            tuple of (AnnotatedDocument, total_annotations_in_file, excluded_class_4_count)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Find matching label if not supplied
        if label_path is None:
            candidate_label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            if candidate_label.exists():
                label_path = candidate_label

        fields: list[AnnotatedField] = []
        total_anns = 0
        excluded_c4 = 0

        if label_path and label_path.exists():
            lines = [l.strip() for l in label_path.read_text("utf-8-sig").splitlines() if l.strip()]
            for line in lines:
                tokens = line.split()
                if not tokens:
                    continue

                total_anns += 1
                try:
                    cls_id = int(tokens[0])
                    coords = [float(x) for x in tokens[1:]]
                except ValueError as exc:
                    raise ValueError(f"Malformed annotation line in {label_path}: '{line}'") from exc

                # Check class mapping
                if cls_id not in self.class_mapping:
                    raise ValueError(
                        f"Unmapped class ID {cls_id} in {label_path} for {self.document_type.value}"
                    )

                field_name = self.class_mapping[cls_id]

                # If class is explicitly mapped to None (e.g. Aadhaar class 4)
                if field_name is None:
                    if cls_id == 4:
                        excluded_c4 += 1
                    continue

                # Validate field name against document vocabulary
                if not is_valid_field(self.document_type, field_name):
                    raise ValueError(
                        f"Field '{field_name}' (class {cls_id}) is not a valid semantic field for {self.document_type.value}"
                    )

                # Parse geometry (box or polygon)
                if len(coords) == 4:
                    cx, cy, w, h = coords
                    geometry = FieldGeometry.from_yolo_box(cx, cy, w, h, is_normalized=True)
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    geometry = FieldGeometry.from_yolo_polygon(coords, is_normalized=True)
                else:
                    raise ValueError(
                        f"Invalid coordinate count ({len(coords)}) in {label_path}: '{line}'"
                    )

                # OCR transcription defaults to MISSING for raw/processed datasets
                transcription = FieldTranscription(
                    text=None,
                    status=TranscriptionStatus.MISSING,
                )

                fields.append(
                    AnnotatedField(
                        field_name=field_name,
                        geometry=geometry,
                        transcription=transcription,
                        original_class_id=cls_id,
                    )
                )

        # Build provenance
        manifest_row = self._manifest_lookup.get(image_path.stem, {})
        split_name = source_split or manifest_row.get("target_split") or image_path.parent.parent.name
        provenance = DocumentProvenance(
            source_dataset=self.document_type.value,
            source_split=split_name,
            source_group_id=manifest_row.get("source_group_id"),
            source_image_rel_path=image_path.as_posix(),
            sha256=manifest_row.get("sha256"),
        )

        doc = AnnotatedDocument(
            image_path=image_path.as_posix(),
            document_type=self.document_type,
            image_size=(width, height),
            fields=fields,
            provenance=provenance,
        )

        # Verify document integrity
        doc.validate_fields(strict=True)

        return doc, total_anns, excluded_c4

    def convert(self, source_dir: Path) -> Iterator[AnnotatedDocument]:
        """Yield one AnnotatedDocument per image in the dataset directory."""
        source_dir = Path(source_dir)
        image_paths = sorted(source_dir.glob("*/**/images/*.*"))
        for img_path in image_paths:
            doc, _, _ = self.convert_file(img_path)
            yield doc

    def convert_dataset(self, source_dir: Path) -> tuple[list[AnnotatedDocument], ConversionStats]:
        """Convert all images and annotations in source_dir, returning documents and stats."""
        source_dir = Path(source_dir)
        stats = ConversionStats(document_type=self.document_type.value)
        documents: list[AnnotatedDocument] = []

        image_paths = sorted(source_dir.glob("*/**/images/*.*"))
        stats.input_images = len(image_paths)

        for img_path in image_paths:
            try:
                doc, total_anns, excluded_c4 = self.convert_file(img_path)
                documents.append(doc)
                stats.converted_documents += 1
                stats.input_annotations += total_anns
                stats.included_fields += len(doc.fields)
                stats.excluded_class_4_annotations += excluded_c4
            except Exception as exc:
                stats.conversion_errors.append(f"{img_path.name}: {type(exc).__name__}: {exc}")

        return documents, stats


def save_documents_jsonl(documents: Sequence[AnnotatedDocument], output_path: str | Path) -> Path:
    """Save a collection of AnnotatedDocuments to a JSON Lines file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict()) + "\n")
    return output_path


def load_documents_jsonl(input_path: str | Path) -> list[AnnotatedDocument]:
    """Load AnnotatedDocuments from a JSON Lines file."""
    input_path = Path(input_path)
    documents: list[AnnotatedDocument] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                documents.append(AnnotatedDocument.from_dict(data))
    return documents
