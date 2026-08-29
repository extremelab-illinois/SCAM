#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Remove generated output files from the examples directory.

Deletes:
  - *.png  (comparison and verification plots)
  - results_*/  (CSV output directories)
  - __pycache__/  (bytecode cache)
  - .~lock.*  (LibreOffice lock files)

Run from anywhere:
    python3 examples/clean_outputs.py
    python3 examples/clean_outputs.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def collect(dry_run: bool) -> None:
    removed: list[Path] = []

    # PNG plots
    for p in sorted(HERE.rglob("*.png")):
        removed.append(p)
        if not dry_run:
            p.unlink()

    # results_* directories
    for p in sorted(HERE.rglob("results_*")):
        if p.is_dir():
            removed.append(p)
            if not dry_run:
                shutil.rmtree(p)

    # __pycache__ directories
    for p in sorted(HERE.rglob("__pycache__")):
        if p.is_dir():
            removed.append(p)
            if not dry_run:
                shutil.rmtree(p)

    # LibreOffice lock files
    for p in sorted(HERE.rglob(".~lock.*")):
        removed.append(p)
        if not dry_run:
            p.unlink()

    if not removed:
        print("Nothing to clean.")
        return

    label = "[dry-run] would remove" if dry_run else "removed"
    for p in removed:
        print(f"  {label}: {p.relative_to(HERE.parent)}")
    print(f"\n{'Would remove' if dry_run else 'Removed'} {len(removed)} item(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be deleted without deleting")
    args = parser.parse_args()
    collect(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
