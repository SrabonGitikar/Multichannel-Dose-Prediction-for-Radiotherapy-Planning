"""
tag_rtdose.py
=============
Walk a testdata directory, find all RTDOSE DICOM files, and update their
SeriesDescription based on the file's last-modified date:

  Modified on June 17  →  "Predicted - 03 june model"
  Modified on June 18  →  "Predicted - 17 june model"

Usage (run from 01 ICON/):
    python utils/tag_rtdose.py
    python utils/tag_rtdose.py --testdata-dir testdata --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pydicom

# Map mday → SeriesDescription
_LABEL = {
    17: "Predicted - 03 june model",
    18: "Predicted - 17 june model",
}


def tag_rtdose_files(testdata_dir: Path, dry_run: bool = False) -> None:
    found = changed = skipped = 0

    for dcm_path in sorted(testdata_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception as e:
            print(f"  [SKIP] Cannot read {dcm_path.name}: {e}")
            skipped += 1
            continue

        modality = getattr(ds, "Modality", "").strip().upper()
        if modality != "RTDOSE":
            continue

        found += 1
        mtime = datetime.fromtimestamp(dcm_path.stat().st_mtime)
        label = _LABEL.get(mtime.day)

        if label is None:
            print(f"  [SKIP] {dcm_path.name}  mtime={mtime.strftime('%Y-%m-%d')}  (no rule for day {mtime.day})")
            skipped += 1
            continue

        current_sd = getattr(ds, "SeriesDescription", "")
        status = "UNCHANGED" if current_sd == label else "UPDATE"
        print(
            f"  [{status}] {dcm_path.relative_to(testdata_dir)}\n"
            f"            mtime={mtime.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"SeriesDescription: {repr(current_sd)} → {repr(label)}"
        )

        if status == "UPDATE" and not dry_run:
            ds = pydicom.dcmread(str(dcm_path))  # reload with pixels for safe save
            ds.SeriesDescription = label
            ds.save_as(str(dcm_path))
            changed += 1

    print(
        f"\nDone. RTDOSE found={found}  updated={changed}  skipped={skipped}"
        + ("  [DRY RUN — no files written]" if dry_run else "")
    )


if __name__ == "__main__":
    _default_testdata = Path(__file__).resolve().parent.parent / "testdata"

    parser = argparse.ArgumentParser(description="Tag RTDOSE DICOMs by creation date")
    parser.add_argument(
        "--testdata-dir", type=Path, default=_default_testdata,
        help=f"Root directory to search (default: {_default_testdata})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without modifying any files",
    )
    args = parser.parse_args()

    if not args.testdata_dir.is_dir():
        print(f"ERROR: directory not found: {args.testdata_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {args.testdata_dir}")
    print(f"Rules:    Jun 17 → '{_LABEL[17]}'")
    print(f"          Jun 18 → '{_LABEL[18]}'")
    print(f"Dry run:  {args.dry_run}\n")

    tag_rtdose_files(args.testdata_dir, dry_run=args.dry_run)
