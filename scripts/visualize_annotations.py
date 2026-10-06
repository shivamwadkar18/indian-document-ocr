"""Local-only annotation visualization for manual class identification.

Usage (Windows):
    .venv\\Scripts\\python scripts\\visualize_annotations.py --dataset aadhaar --split train --samples-per-class 5
    .venv\\Scripts\\python scripts\\visualize_annotations.py --dataset pan --split all

Reads data/raw/<dataset>/<split>/{images,labels} read-only and writes contact
sheets + index.json to data/reports/visualizations/<dataset>/<split>/, then
rebuilds class_identification_report.md. Class IDs are shown as "class=<id>";
meanings are not inferred. Nothing is uploaded and no OCR is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from idocr.data.audit.layout import detect_layout
from idocr.data.visualization.workflow import VisualizationError, check_output_dir, visualize_split, write_report
from idocr.utils import ProjectPaths, load_default_config


def main() -> int:
    config = load_default_config()
    paths = ProjectPaths.from_config(config)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=config["documents"])
    parser.add_argument("--split", default="all", help="train | valid | test | all (default: all)")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--mixed-samples", type=int, default=12)
    parser.add_argument("--anomaly-samples", type=int, default=None, help="default: --samples-per-class")
    parser.add_argument("--output-dir", type=Path, default=paths.reports / "visualizations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=640, help="max tile side in px (downscale only)")
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    dataset_root = paths.raw_dir(args.dataset)
    available = [s.name for s in detect_layout(dataset_root).splits] if dataset_root.is_dir() else []
    if not available:
        print(f"No splits found under {dataset_root}", file=sys.stderr)
        return 1
    splits = available if args.split == "all" else [args.split]
    missing = [s for s in splits if s not in available]
    if missing:
        print(f"Split(s) {missing} not present for {args.dataset}; available: {available}", file=sys.stderr)
        return 1

    try:
        check_output_dir(args.output_dir, [paths.raw])
    except VisualizationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for split in splits:
        index = visualize_split(
            args.dataset, dataset_root, split, args.output_dir,
            samples_per_class=args.samples_per_class, mixed_samples=args.mixed_samples,
            anomaly_samples=args.anomaly_samples, seed=args.seed, tile_size=args.tile_size, columns=args.columns,
        )
        print(f"{args.dataset}/{split}: {index['images']} images -> {len(index['sheets'])} sheets in "
              f"{args.output_dir / args.dataset / split}")
        for s in index["sheets"]:
            print(f"  {s['file']:<45} {len(s['tiles'])} tiles")
        not_found = [k for k, a in index["anomalies"].items() if not a["found"]]
        if not_found:
            print(f"  anomalies not found: {', '.join(not_found)}")
    print(f"Report: {write_report(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
