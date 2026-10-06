"""Run the read-only dataset audit and write reports to data/reports/.

Usage:
    python scripts/audit_dataset.py --document aadhaar
    python scripts/audit_dataset.py --path data/raw/pan --name pan

Writes <name>_report.json and regenerates dataset_summary.md from every
*_report.json in the reports directory. The dataset is never modified: every
source file is hashed before and after, and the run fails if anything changed.
Label previews in reports are redacted; reports stay local (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from idocr.data.audit import AuditError, run_audit, write_summary
from idocr.utils import ProjectPaths, load_default_config


def main() -> int:
    config = load_default_config()
    paths = ProjectPaths.from_config(config)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document", choices=config["documents"], help="audit data/raw/<document>")
    group.add_argument("--path", type=Path, help="audit an arbitrary dataset directory")
    parser.add_argument("--name", help="report name (default: document / directory name)")
    parser.add_argument("--reports-dir", type=Path, default=paths.reports)
    args = parser.parse_args()

    dataset_dir = paths.raw_dir(args.document) if args.document else args.path
    try:
        report = run_audit(dataset_dir, args.reports_dir, name=args.name or args.document, config=config.get("audit"))
    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    summary = write_summary(args.reports_dir)
    integ = report.sections.get("source_integrity")
    print(f"Audited {report.dataset_root}")
    print(f"  report : {args.reports_dir / (report.dataset_name + '_report.json')}")
    print(f"  summary: {summary}")
    print(f"  issues : {report.sections['issue_counts'] or 'none'}")
    if integ:
        print(f"  SOURCE DATA MODIFIED: {'YES' if integ['source_data_modified'] else 'NO'} "
              f"({integ['files_checked']} files hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
