"""Run the Aadhaar/PAN pipeline on one image and print the result as JSON.

Usage:
    .venv\Scripts\python scripts\run_inference.py --image path\to\card.jpg
    python scripts/run_inference.py --image card.jpg --document-type pan \
        --detector-weights /content/drive/MyDrive/idocr/detector/best.pt \
        --recognizer-checkpoint /content/drive/MyDrive/idocr/recognizer/best.pt --device cuda

Asset paths and options come from configs/inference.yaml; flags override them.
The output contains the document's personal data; handle it accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from idocr.pipeline import DocumentPipeline, PipelineConfigurationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--detector-weights")
    parser.add_argument("--recognizer-checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--document-type", choices=["auto", "aadhaar", "pan"], default="auto")
    args = parser.parse_args()

    try:
        pipeline = DocumentPipeline.from_config(
            args.config,
            detector_weights=args.detector_weights,
            recognizer_checkpoint=args.recognizer_checkpoint,
            device=args.device,
        )
        result = pipeline.run_file(args.image, None if args.document_type == "auto" else args.document_type)
    except (FileNotFoundError, ValueError, TypeError, PipelineConfigurationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
