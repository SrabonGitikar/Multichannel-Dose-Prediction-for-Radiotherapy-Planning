#!/usr/bin/env python3
"""
run_dvh_batch.py
================
Extracts the PRIME DVH validation ZIP archive and runs the DVH comparison
(Eclipse vs AI) on all patients it contains, producing per-patient CSVs
and a single combined summary CSV.

Usage:
    python run_dvh_batch.py [--zip PATH] [--extract-to DIR] [--output-dir DIR]

Defaults:
    --zip         "01 ICON/utils/17.06.2026 - PRIME DVH validation-20260617T090008Z-3-001.zip"
    --extract-to  dvh_data/              (created in the project root)
    --output-dir  dvh_results/           (created in the project root)
"""

import os
import io
import sys
import shutil
import zipfile
import argparse
import tempfile
from pathlib import Path

import pydicom
import pandas as pd

# ── Ensure dicompylercore is importable ──────────────────────────────────────
_SITE_PKG = Path("/home/ankit/Dose_pred/dose_env/lib/python3.12/site-packages")
if _SITE_PKG.exists() and str(_SITE_PKG) not in sys.path:
    sys.path.insert(0, str(_SITE_PKG))

# ── Import DVH functions from sibling script ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate_dvh_comparison import (   # noqa: E402
    scan_dicom_dir,
    calculate_dvh_metrics,
    build_dataframe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract the nested ZIP
# ─────────────────────────────────────────────────────────────────────────────

def extract_archive(zip_path: str, extract_to: str) -> Path:
    """
    The upload is a doubly-nested ZIP:
      outer.zip
        └── 17.06.2026 - PRIME DVH validation/
                └── testdata.zip
                        └── testdata/
                                ├── <patient-uuid>/
                                │       └── <series-uid>/
                                │               └── *.dcm
                                └── ...

    Extracts everything into extract_to/testdata/ and returns that path.
    If it already exists (re-run), skips extraction.
    """
    extract_to = Path(extract_to)
    testdata_dir = extract_to / "testdata"

    if testdata_dir.exists() and any(testdata_dir.iterdir()):
        print(f"[extract] Already extracted → {testdata_dir}  (skipping)")
        return testdata_dir

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[extract] Opening outer ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path) as z_outer:
        inner_name = next(
            n for n in z_outer.namelist() if n.endswith("testdata.zip")
        )
        print(f"[extract] Reading inner ZIP: {inner_name}")
        inner_bytes = z_outer.read(inner_name)

    print(f"[extract] Extracting {len(inner_bytes)/1e6:.1f} MB inner archive ...")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z_inner:
        members = [m for m in z_inner.infolist() if not m.filename.endswith("/")]
        total   = len(members)
        for i, member in enumerate(members, 1):
            z_inner.extract(member, extract_to)
            if i % 200 == 0 or i == total:
                print(f"  {i}/{total} files extracted", end="\r")
    print()
    print(f"[extract] Done → {testdata_dir}")
    return testdata_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Flatten each patient folder into a single directory
# ─────────────────────────────────────────────────────────────────────────────

def collect_patient_dcm_files(patient_root: Path) -> list:
    """
    Recursively collect all .dcm files under a patient folder regardless
    of the nested series-UID sub-directory layout.
    """
    return list(patient_root.rglob("*"))


def flatten_patient(patient_root: Path, flat_dir: Path) -> Path:
    """
    Symlink all DICOM files from the nested series sub-dirs into a single
    flat directory so that scan_dicom_dir() can find them in one rglob pass.
    Returns flat_dir (created if necessary).
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    for src in patient_root.rglob("*"):
        if src.is_file():
            dst = flat_dir / src.name
            if not dst.exists():
                # Use hard link for speed (no copy overhead)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    return flat_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Process all patients
# ─────────────────────────────────────────────────────────────────────────────

def process_all_patients(testdata_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Iterate over every patient sub-directory, run the DVH comparison,
    save per-patient CSVs, and return the combined DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    patient_dirs = sorted([d for d in testdata_dir.iterdir() if d.is_dir()])
    print(f"\n[batch] Found {len(patient_dirs)} patient directories\n")

    for idx, pat_dir in enumerate(patient_dirs, 1):
        print("=" * 70)
        print(f"  Patient {idx}/{len(patient_dirs)}: {pat_dir.name}")
        print("=" * 70)

        # Flatten nested series dirs into one temp folder for easy scanning
        flat_dir = output_dir / "tmp_flat" / pat_dir.name
        flatten_patient(pat_dir, flat_dir)

        try:
            # ── Identify files ─────────────────────────────────────────────
            rtstruct_path, dose_eclipse, dose_ai, patient_id = \
                scan_dicom_dir(str(flat_dir))

            # ── Eclipse DVH ────────────────────────────────────────────────
            print(f"\n  → Eclipse DVH")
            eclipse_metrics = calculate_dvh_metrics(rtstruct_path, dose_eclipse)

            # ── AI DVH ────────────────────────────────────────────────────
            print(f"\n  → AI DVH")
            ai_metrics = calculate_dvh_metrics(rtstruct_path, dose_ai)

            # ── Build DataFrame ───────────────────────────────────────────
            df = build_dataframe(eclipse_metrics, ai_metrics)
            df.insert(0, "PatientID", patient_id)   # prepend patient column

            # ── Print patient table ────────────────────────────────────────
            print(f"\n  DVH Comparison — {patient_id}")
            print(df.drop(columns="PatientID").to_string(index=False))

            # ── Save per-patient CSV ──────────────────────────────────────
            safe_id = re.sub(r"[^\w\-.]", "_", patient_id)[:40]
            per_csv = output_dir / f"{safe_id}_dvh_comparison.csv"
            df.to_csv(per_csv, index=False)
            print(f"\n  Saved → {per_csv.name}")

            all_dfs.append(df)

        except Exception as exc:
            print(f"  [ERROR] Patient {pat_dir.name}: {exc}")
            import traceback; traceback.print_exc()

        print()

    # ── Clean up temp flat dirs ──────────────────────────────────────────────
    tmp_flat = output_dir / "tmp_flat"
    if tmp_flat.exists():
        shutil.rmtree(tmp_flat)

    if not all_dfs:
        print("[batch] No results collected.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(combined: pd.DataFrame, output_dir: Path):
    """
    Print and save a cohort-level summary: mean ± std per ROI × Metric.
    """
    if combined.empty:
        return

    print("\n" + "=" * 70)
    print("  COHORT SUMMARY  (mean ± std across all patients)")
    print("=" * 70)

    summary_rows = []
    for (roi, metric), grp in combined.groupby(["ROI", "Metric"]):
        row = {
            "ROI"             : roi,
            "Metric"          : metric,
            "Eclipse_mean"    : round(grp["Eclipse_Dose"].mean(), 2),
            "Eclipse_std"     : round(grp["Eclipse_Dose"].std(), 2),
            "Model_mean"      : round(grp["Model_Dose"].mean(), 2),
            "Model_std"       : round(grp["Model_Dose"].std(), 2),
            "N_patients"      : grp["Eclipse_Dose"].notna().sum(),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    combined_csv = output_dir / "ALL_patients_dvh_comparison.csv"
    summary_csv  = output_dir / "ALL_patients_dvh_summary.csv"
    combined.to_csv(combined_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[batch] All-patient table  → {combined_csv}")
    print(f"[batch] Cohort summary     → {summary_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

import re   # needed by process_all_patients


def main():
    _proj_root = Path(__file__).resolve().parent
    default_zip = str(
        _proj_root
        / "01 ICON" / "utils"
        / "17.06.2026 - PRIME DVH validation-20260617T090008Z-3-001.zip"
    )

    parser = argparse.ArgumentParser(
        description="Batch DVH comparison: Eclipse vs AI for all PRIME patients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zip",         default=default_zip,
                        help="Path to the outer validation ZIP file")
    parser.add_argument("--extract-to",  default=str(_proj_root / "dvh_data"),
                        help="Directory to extract the DICOM archive into")
    parser.add_argument("--output-dir",  default=str(_proj_root / "dvh_results"),
                        help="Directory for per-patient and combined CSVs")
    args = parser.parse_args()

    # 1. Extract
    testdata_dir = extract_archive(args.zip, args.extract_to)

    # 2. Process all patients
    combined = process_all_patients(Path(testdata_dir), Path(args.output_dir))

    # 3. Cohort summary
    print_summary(combined, Path(args.output_dir))


if __name__ == "__main__":
    main()
