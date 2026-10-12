"""Private real-crop OCR benchmark: create, review status, transcribe, evaluate.

The benchmark directory must be private (e.g. Google Drive, or a git-ignored
path under data/). Output here shows opaque sample IDs and counts only.

Usage:
    python scripts/real_crop_benchmark.py create   --benchmark-dir DIR --name real_crops_v1 \\
        [--dataset-yaml data/processed/detector/data.yaml] [--seed 42] [--target 40]
    python scripts/real_crop_benchmark.py status   --benchmark-dir DIR
    python scripts/real_crop_benchmark.py transcribe --benchmark-dir DIR      # local terminal
    python scripts/real_crop_benchmark.py evaluate --benchmark-dir DIR --checkpoint best.pt \\
        --run-name crnn_v4_ep8 [--device cuda]

In Colab/Jupyter, transcribe from a notebook cell instead (stdin of `!python`
is not interactive):
    from idocr.evaluation.real_benchmark import colab_transcribe
    colab_transcribe("/content/drive/MyDrive/idocr/benchmarks/real_crops_v1")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from idocr.evaluation.real_benchmark import (
    DEFAULT_TARGET_PER_FIELD,
    BenchmarkError,
    ReferenceStore,
    create_benchmark,
    evaluate_benchmark,
    format_report,
    load_manifest,
    progress,
    transcribe,
)


def _print_progress(benchmark_dir: Path) -> None:
    manifest = load_manifest(benchmark_dir)
    prog = progress(manifest, ReferenceStore(benchmark_dir).records())
    print(f"{'field':15s} {'verified':>8s} {'unclear':>8s} {'skipped':>8s} {'pending':>8s}  status")
    for field, p in prog.items():
        state = "complete" if p["complete"] else ("EXHAUSTED" if p["exhausted"] else "in progress")
        print(f"{field:15s} {p['verified']:8d} {p['unclear']:8d} {p['skipped']:8d} {p['pending']:8d}  "
              f"{state} (target {p['target']})")


def cmd_create(args: argparse.Namespace) -> None:
    from idocr.training.detection.dataset import resolve_dataset

    dataset = resolve_dataset(args.dataset_yaml)
    if "test" not in dataset.splits:
        raise BenchmarkError(f"{args.dataset_yaml} defines no test split")
    manifest = create_benchmark(
        dataset.splits["test"], args.benchmark_dir, name=args.name, seed=args.seed,
        target_per_field=args.target, source_description="detector test split, ground-truth boxes",
    )
    print(f"Created benchmark {manifest['name']} (seed {manifest['seed']}).")
    print(f"{'field':15s} {'candidates':>10s} {'documents':>10s}")
    for field, queue in manifest["queues"].items():
        print(f"{field:15s} {len(queue):10d} {manifest['available_documents'][field]:10d}")
    if manifest["excluded"]:
        print(f"Excluded boxes: {manifest['excluded']}")


def cmd_transcribe(args: argparse.Namespace) -> None:
    def show(image, caption):  # opens the system image viewer
        print(caption)
        image.show()

    transcribe(args.benchmark_dir, ask=input, show=show, max_items=args.max_items)
    _print_progress(args.benchmark_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from idocr.models.recognizer.crnn import CRNNRecognizer

    checkpoint = Path(args.checkpoint)
    recognizer = CRNNRecognizer.from_checkpoint(checkpoint, device=args.device)
    model_info = {
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "alphabet_size": len(recognizer.vocab.alphabet),
    }
    report = evaluate_benchmark(args.benchmark_dir, recognizer, run_name=args.run_name, model_info=model_info)
    print(format_report(report))
    print(f"\nAggregate report: reports/{args.run_name}.json (metrics only)")
    if args.json:
        print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset-yaml", type=Path, default=Path("data/processed/detector/data.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET_PER_FIELD)
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("status")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.set_defaults(func=lambda a: _print_progress(a.benchmark_dir))

    p = sub.add_parser("transcribe")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int)
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("evaluate")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true", help="also print the aggregate JSON (metrics only)")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    try:
        args.func(args)
    except (BenchmarkError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
