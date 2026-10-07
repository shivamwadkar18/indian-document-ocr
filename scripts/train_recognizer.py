"""Train the CRNN OCR recognizer from a config in configs/recognition/.

Usage:
    .venv\Scripts\python scripts\train_recognizer.py --config configs\recognition\crnn_smoke_cpu.yaml
    python scripts/train_recognizer.py --config configs/recognition/crnn_smoke_gpu.yaml \
        --experiment-name smoke_test_crnn_t4_2 --epochs 5

CLI flags override the YAML. Runs go to <output_dir>/<experiment_name>/ and an
existing run directory is never overwritten. After training, best.pt is
reloaded through CRNNRecognizer and used to decode a few test crops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from idocr.models.recognizer.crnn import CRNNRecognizer
from idocr.training.detection.train import RunExistsError
from idocr.training.recognition import RecognizerConfigError, load_recognizer_config, train
from idocr.training.recognition.data import read_split


def decode_test_samples(run_dir: Path, dataset_dir: Path, device: str, n: int) -> None:
    recognizer = CRNNRecognizer.from_checkpoint(run_dir / "checkpoints" / "best.pt", device=device)
    test_dir = dataset_dir / "test"
    samples = read_split(test_dir, n)
    # Dataset images are already preprocessed. Strip the padding so that
    # recognize()'s own preprocessing reproduces the stored image exactly.
    h_pad = recognizer.preprocessing["horizontal_padding"]
    v_pad = recognizer.preprocessing["vertical_padding"]
    crops = [np.asarray(Image.open(test_dir / s.image_path).convert("L"))[v_pad:-v_pad or None, h_pad:-h_pad or None]
             for s in samples]
    print("\nReloaded best.pt via CRNNRecognizer; decoding test crops:")
    for s, r in zip(samples, recognizer.recognize_batch(crops)):
        mark = "OK " if r.text == s.text else "   "
        print(f"  {mark}{s.field_name:15s} ref={s.text!r:20s} pred={r.text!r:20s} conf={r.confidence:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", dest="experiment_name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--max-train-samples", dest="max_train_samples", type=int)
    parser.add_argument("--max-valid-samples", dest="max_valid_samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decode-samples", type=int, default=12)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    decode_n = args.pop("decode_samples")

    try:
        cfg = load_recognizer_config(config_path, overrides=args)
        record = train(cfg)
    except (RecognizerConfigError, RunExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({k: record[k] for k in ("experiment_name", "status", "device", "run_dir", "checkpoints",
                                             "best_epoch", "best_valid_cer", "final_valid")}, indent=2))
    if decode_n > 0:
        dataset_dir = Path(record["dataset"]["dir"])
        decode_test_samples(Path(record["run_dir"]), dataset_dir, record["device"], decode_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
