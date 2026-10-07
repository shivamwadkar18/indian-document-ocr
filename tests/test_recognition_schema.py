import pytest

from idocr.data.recognition import RecognitionSample


def test_valid_recognition_sample() -> None:
    sample = RecognitionSample(
        image_path="sample.png",
        text="ABCDE1234F",
        field_name="pan_number",
        document_type="pan",
        source="synthetic",
        split="train",
    )

    result = sample.to_dict()

    assert result["text"] == "ABCDE1234F"
    assert result["field_name"] == "pan_number"
    assert result["source"] == "synthetic"
    assert result["split"] == "train"


def test_empty_text_rejected() -> None:
    with pytest.raises(ValueError):
        RecognitionSample(
            image_path="sample.png",
            text="",
            field_name="name",
            document_type="aadhaar",
            source="synthetic",
            split="train",
        )


def test_invalid_source_rejected() -> None:
    with pytest.raises(ValueError):
        RecognitionSample(
            image_path="sample.png",
            text="TEST",
            field_name="name",
            document_type="aadhaar",
            source="unknown",
            split="train",
        )


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError):
        RecognitionSample(
            image_path="sample.png",
            text="TEST",
            field_name="name",
            document_type="aadhaar",
            source="synthetic",
            split="production",
        )
