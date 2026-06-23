"""
copy_test_cases.py
------------------
Copy DICOM case directories listed in a validation CSV into a new folder.

Usage
-----
python utils/copy_test_cases.py \
    --src-dir  /path/to/original/dicom/root \
    --csv      /path/to/validation.csv \
    --dst-dir  /path/to/new/output/folder \
    [--id-col  Patient_ID]          # CSV column name (default: Patient_ID)
    [--match   exact|contains]      # matching strategy (default: exact)
    [--dry-run]                     # print plan without copying
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


def load_patient_ids(csv_path: str, id_col: str) -> list:
    ids = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{id_col}' not found in CSV. "
                f"Available columns: {list(reader.fieldnames)}"
            )
        for row in reader:
            pid = row[id_col].strip()
            if pid:
                ids.append(pid)
    return ids


def _hex_fragments(patient_id: str) -> list:
    """Extract all underscore/dash-delimited hex fragments from a patient ID.

    e.g. 'prostate_f40d8cf2_f05' → ['prostate', 'f40d8cf2', 'f05']
    We return the fragments that look like hex strings (all hex chars).
    """
    parts = re.split(r"[_\-]", patient_id)
    return [p for p in parts if p and all(c in "0123456789abcdefABCDEF" for c in p)]


def find_case_dir(src_root: Path, patient_id: str, match: str) -> Path | None:
    """Return the first subdirectory in src_root that matches patient_id."""
    hex_frags = _hex_fragments(patient_id)

    for candidate in src_root.iterdir():
        if not candidate.is_dir():
            continue
        name = candidate.name

        if match == "exact" and name == patient_id:
            return candidate

        if match == "contains":
            # 1. Direct substring match
            if patient_id in name:
                return candidate
            # 2. Smart hex-prefix match:
            #    folder names like 'f40d8cf2.60a5.4b9e...' share the first
            #    hex group with patient IDs like 'prostate_f40d8cf2_f05'.
            #    Try matching any hex fragment against the start of the folder name
            #    or against any dot-separated segment of the folder name.
            folder_parts = name.replace(".", "_").split("_")
            for frag in hex_frags:
                if any(fp.lower().startswith(frag.lower()) or
                       frag.lower().startswith(fp.lower())
                       for fp in folder_parts):
                    return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Copy DICOM case folders listed in a CSV to a new directory"
    )
    parser.add_argument("--src-dir", required=True, type=str,
                        help="Root directory containing one subdirectory per case")
    parser.add_argument("--csv", required=True, type=str,
                        help="Path to the validation CSV file")
    parser.add_argument("--dst-dir", required=True, type=str,
                        help="Destination directory (created if it does not exist)")
    parser.add_argument("--id-col", default="Patient_ID", type=str,
                        help="CSV column name containing patient IDs (default: Patient_ID)")
    parser.add_argument("--match", default="exact", choices=["exact", "contains"],
                        help="Directory matching strategy: "
                             "'exact' = folder name == patient ID, "
                             "'contains' = folder name contains patient ID "
                             "(default: exact)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the copy plan without actually copying anything")
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)

    if not src_root.is_dir():
        print(f"[ERROR] Source directory not found: {src_root}")
        sys.exit(1)

    # ── Load patient IDs from CSV ─────────────────────────────────────────────
    try:
        patient_ids = load_patient_ids(args.csv, args.id_col)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nCSV       : {args.csv}")
    print(f"Source    : {src_root}")
    print(f"Dest      : {dst_root}")
    print(f"ID column : {args.id_col}")
    print(f"Match     : {args.match}")
    print(f"Cases     : {len(patient_ids)}")
    if args.dry_run:
        print("Mode      : DRY RUN (nothing will be copied)\n")
    else:
        print()

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    # ── Copy / report ─────────────────────────────────────────────────────────
    found, not_found, skipped = [], [], []

    for pid in patient_ids:
        case_dir = find_case_dir(src_root, pid, args.match)
        if case_dir is None:
            not_found.append(pid)
            print(f"  [NOT FOUND] {pid}")
            continue

        dst_case = dst_root / case_dir.name
        if dst_case.exists():
            skipped.append(pid)
            print(f"  [SKIP]      {pid}  →  {dst_case}  (already exists)")
            continue

        found.append(pid)
        print(f"  [COPY]      {pid}  →  {dst_case}")
        if not args.dry_run:
            shutil.copytree(str(case_dir), str(dst_case))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Total in CSV  : {len(patient_ids)}")
    print(f"  Copied        : {len(found)}")
    print(f"  Skipped (dup) : {len(skipped)}")
    print(f"  Not found     : {len(not_found)}")
    if not_found:
        print(f"\n  Missing cases:")
        for pid in not_found:
            print(f"    - {pid}")


if __name__ == "__main__":
    main()
