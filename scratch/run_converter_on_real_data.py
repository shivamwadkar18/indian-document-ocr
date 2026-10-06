"""Run YoloDatasetConverter on processed Aadhaar and PAN datasets and print statistics."""

from pathlib import Path
from idocr.data.converters import YoloDatasetConverter
from idocr.types import DocumentType

def run_conversion():
    aadhaar_dir = Path("data/processed/aadhaar")
    aadhaar_manifest = Path("data/processed/manifests/aadhaar_source_groups.csv")
    pan_dir = Path("data/processed/pan")
    pan_manifest = Path("data/processed/manifests/pan_source_groups.csv")
    
    print("==================================================")
    print("CONVERTING PROCESSED AADHAAR DATASET")
    print("==================================================")
    aadhaar_conv = YoloDatasetConverter(
        document_type=DocumentType.AADHAAR,
        manifest_path=aadhaar_manifest,
    )
    aadhaar_docs, aadhaar_stats = aadhaar_conv.convert_dataset(aadhaar_dir)
    print(f"Aadhaar conversion stats:")
    for k, v in aadhaar_stats.to_dict().items():
        print(f"  {k}: {v}")
        
    print("\n==================================================")
    print("CONVERTING PROCESSED PAN DATASET")
    print("==================================================")
    pan_conv = YoloDatasetConverter(
        document_type=DocumentType.PAN,
        manifest_path=pan_manifest,
    )
    pan_docs, pan_stats = pan_conv.convert_dataset(pan_dir)
    print(f"PAN conversion stats:")
    for k, v in pan_stats.to_dict().items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_conversion()
