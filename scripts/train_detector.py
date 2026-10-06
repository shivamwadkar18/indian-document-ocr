"""Train the field detector from a config in configs/detector/.

Usage:
    .venv\\Scripts\\python scripts\\train_detector.py --config configs\\detector\\smoke_test_yolov8n.yaml
    .venv\\Scripts\\python scripts\\train_detector.py --config configs\\detector\\smoke_test_yolov8n.yaml \\
        --experiment-name smoke_test_yolov8n_2 --fraction 0.1

CLI flags override the YAML. Runs go to <output_dir>/<experiment_name>/ and an
existing run directory is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from idocr.training.detection import DetectorConfigError, DetectorDatasetError, RunExistsError, load_detector_config, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", dest="experiment_name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    args = vars(parser.parse_args())
    config_path = args.pop("config")

    try:
        cfg = load_detector_config(config_path, overrides=args)
        record = train(cfg)
    except (DetectorConfigError, DetectorDatasetError, RunExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: record[k] for k in ("experiment_name", "status", "checkpoints")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
