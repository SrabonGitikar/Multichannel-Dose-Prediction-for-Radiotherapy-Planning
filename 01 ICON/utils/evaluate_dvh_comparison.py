#!/usr/bin/env python3
"""
evaluate_dvh_comparison.py
==========================
Compare Eclipse (clinical) vs AI-predicted RTDOSE using DVH metrics.

Given a patient DICOM directory containing:
  - CT slices
  - 1 × RTSTRUCT
  - 1 × RTPLAN  (optional)
  - 2 × RTDOSE  (one Eclipse, one AI with SeriesDescription == "Predicted Dose (AI)")

Outputs:
  - Side-by-side Pandas DataFrame: ROI | Metric | Eclipse_Dose | Model_Dose
  - CSV file:  {patient_id}_dvh_comparison.csv

Usage:
    python evaluate_dvh_comparison.py /path/to/patient/dicom_dir [--output-dir .]

Requirements (install into your project venv):
    pip install dicompyler-core pandas
"""

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pydicom
import pandas as pd

# ── dicompylercore is installed as 'dicompyler-core'; ensure importable ────────
_SITE_PKG = Path("/home/ankit/Dose_pred/dose_env/lib/python3.12/site-packages")
if _SITE_PKG.exists() and str(_SITE_PKG) not in sys.path:
    sys.path.insert(0, str(_SITE_PKG))

from dicompylercore import dicomparser, dvhcalc   # noqa: E402


# ── Canonical ROI regex mapping  ──────────────────────────────────────────────
#  Patterns tried in order; first match wins.
ROI_PATTERNS = [
    # PTVs — specific prescription levels only (no generic PTV fallback)
    (r"(?i).*PTV.?62.*",          "PTV_62_20"),
    (r"(?i).*PTV.?44.*",          "PTV_44_20"),
    # OARs — bladder / anorectum / bowel / penile
    (r"(?i).*Bladder.*",          "Bladder"),
    (r"(?i).*Anorectum.*",        "Anorectum"),
    (r"(?i).*Rectum.*",           "Anorectum"),    # alias
    (r"(?i).*Bag.?Bowel.*",       "Bag_Bowel"),
    (r"(?i).*Bowel.?Bag.*",       "Bag_Bowel"),
    (r"(?i).*Small.?Bowel.*",     "Bag_Bowel"),
    (r"(?i).*Penile.?Bulb.*",     "Penile_Bulb"),
    (r"(?i).*Penile.*",           "Penile_Bulb"),
    # Femoral heads — handles spaces, underscores, and suffixes like L/Left/R/Right
    (r"(?i).*(Femor(?:al)?|FemHead|Femur).*(?:^|[\W_])L(?:eft)?(?:[\W_]|$)",   "FemHead_L"),
    (r"(?i).*(?:^|[\W_])L(?:eft)?(?:[\W_]|$).*(Femor(?:al)?|FemHead|Femur)",   "FemHead_L"),
    (r"(?i).*(Femor(?:al)?|FemHead|Femur).*(?:^|[\W_])R(?:ight)?(?:[\W_]|$)",  "FemHead_R"),
    (r"(?i).*(?:^|[\W_])R(?:ight)?(?:[\W_]|$).*(Femor(?:al)?|FemHead|Femur)",  "FemHead_R"),
]

# Metrics per canonical ROI
PTV_METRICS     = ["D95"]                      # Gy   — dose covering 95% of volume
OAR_METRICS     = ["V40", "V47", "V53", "V59"] # %pts — volume above dose threshold
PENILE_METRICS  = ["V47"]                      # %pts — volume above 47 Gy
FEMORAL_METRICS = ["Dmax"]                     # Gy   — maximum dose (clinical hard constraint)


def match_roi(roi_name: str):
    """Map a raw RTSTRUCT ROI name to a canonical name, or None."""
    for pattern, canonical in ROI_PATTERNS:
        if re.match(pattern, roi_name):
            return canonical
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Scan directory
# ─────────────────────────────────────────────────────────────────────────────

def scan_dicom_dir(patient_dir: str):
    """
    Scan patient_dir for RTSTRUCT and exactly 2 RTDOSE files.
    Identifies AI vs Eclipse RTDOSE by SeriesDescription.

    Returns
    -------
    rtstruct_path, dose_eclipse_path, dose_ai_path, patient_id
    """
    patient_dir  = Path(patient_dir)
    rtstruct_path = None
    rtdose_files  = []   # list of (path_str, ds_header)
    patient_id    = "unknown"

    for fpath in sorted(patient_dir.rglob("*")):
        if not fpath.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
        except Exception:
            continue

        modality = getattr(ds, "Modality", "")
        if patient_id == "unknown":
            patient_id = str(getattr(ds, "PatientID", "unknown"))

        if modality == "RTSTRUCT" and rtstruct_path is None:
            rtstruct_path = str(fpath)
            print(f"[dvh] RTSTRUCT   : {fpath.name}")
        elif modality == "RTDOSE":
            rtdose_files.append((str(fpath), ds))
            print(f"[dvh] RTDOSE     : {fpath.name}"
                  f"  desc='{getattr(ds, 'SeriesDescription', 'N/A')}'")

    if rtstruct_path is None:
        raise FileNotFoundError(f"No RTSTRUCT found in {patient_dir}")

    if len(rtdose_files) < 2:
        raise ValueError(
            f"Expected ≥ 2 RTDOSE files, found {len(rtdose_files)} in {patient_dir}"
        )

    # Identify AI vs Eclipse
    dose_ai_path      = None
    dose_eclipse_path = None
    for fpath, ds in rtdose_files:
        desc = str(getattr(ds, "SeriesDescription", ""))
        if desc == "Predicted Dose (AI)":
            dose_ai_path = fpath
        elif dose_eclipse_path is None:
            dose_eclipse_path = fpath   # first non-AI dose = Eclipse

    if dose_eclipse_path is None:
        raise ValueError("Could not identify Eclipse RTDOSE.")
    if dose_ai_path is None:
        descs = [(Path(fp).name, getattr(ds, "SeriesDescription", "N/A"))
                 for fp, ds in rtdose_files]
        raise ValueError(
            f"Could not identify AI RTDOSE. SeriesDescriptions: {descs}\n"
            "The AI RTDOSE must have SeriesDescription == 'Predicted Dose (AI)'"
        )

    print(f"[dvh] Patient ID : {patient_id}")
    print(f"[dvh] Eclipse    : {Path(dose_eclipse_path).name}")
    print(f"[dvh] AI         : {Path(dose_ai_path).name}")
    return rtstruct_path, dose_eclipse_path, dose_ai_path, patient_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — DVH calculation engine
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metric(dvh_obj, metric_str: str):
    """
    Extract one metric from a dicompylercore 0.5.7 DVH object.

    dicompyler-core 0.5.7 API:
      dvh.statistic("D95")         → dose (Gy) at 95% volume
      dvh.statistic("V40Gy")       → volume at 40 Gy (on relative_volume DVH → %)
      dvh.max                      → DVHValue with maximum dose

    Supported metric_str values:
      D95   → dose (Gy) covering 95% of the structure volume
      V40   → % of structure volume receiving ≥ 40 Gy
      Dmax  → maximum dose (Gy) anywhere in the structure

    Returns float rounded to 2 dp, or None on failure / empty structure.
    """
    if dvh_obj is None or dvh_obj.volume == 0:
        return None

    try:
        if metric_str == "Dmax":
            # Maximum dose — dvh.max is a DVHValue in dose_units (Gy)
            return round(float(dvh_obj.max.value), 2)

        elif metric_str.startswith("D"):
            # D95 → statistic("D95") returns DVHValue in Gy
            result = dvh_obj.statistic(metric_str)
            return round(float(result.value), 2)

        elif metric_str.startswith("V"):
            # V40 → statistic("V40Gy") on relative_volume DVH returns %
            dose_gy  = metric_str[1:]           # e.g. "40"
            stat_key = f"V{dose_gy}Gy"         # e.g. "V40Gy"
            result   = dvh_obj.relative_volume.statistic(stat_key)
            return round(float(result.value), 2)

    except Exception as exc:
        print(f"    [warn] metric {metric_str}: {exc}")
    return None



def calculate_dvh_metrics(rtstruct_path: str, rtdose_path: str) -> dict:
    """
    Calculate DVH metrics for every matched ROI.

    Returns
    -------
    dict: {canonical_name: {metric_str: float_or_None, ...}}
    """
    rtss_parser = dicomparser.DicomParser(rtstruct_path)
    structures  = rtss_parser.GetStructures()   # {roi_num: {'name':..., 'type':...}}

    results = {}

    for roi_num, roi_info in sorted(structures.items()):
        raw_name  = roi_info.get("name", "")
        canonical = match_roi(raw_name)
        if canonical is None:
            continue

        print(f"  ROI {roi_num:3d}  '{raw_name}'  →  {canonical}")

        try:
            dvh_obj = dvhcalc.get_dvh(rtstruct_path, rtdose_path, roi_num)
        except Exception as exc:
            print(f"    [warn] DVH failed: {exc}")
            dvh_obj = None

        # Determine metrics for this canonical type
        if canonical.startswith("PTV"):
            metrics = PTV_METRICS
        elif canonical == "Penile_Bulb":
            metrics = PENILE_METRICS
        elif canonical in ("FemHead_L", "FemHead_R"):
            metrics = FEMORAL_METRICS
        else:
            metrics = OAR_METRICS

        if canonical not in results:
            results[canonical] = {}

        for metric in metrics:
            # Keep first non-None value if same canonical appears multiple times
            if metric not in results[canonical] or results[canonical][metric] is None:
                results[canonical][metric] = _extract_metric(dvh_obj, metric)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(eclipse_metrics: dict, ai_metrics: dict) -> pd.DataFrame:
    """
    Construct the side-by-side comparison DataFrame.

    Columns: ['ROI', 'Metric', 'Eclipse_Dose', 'Model_Dose',
              'Abs_Error', 'Rel_Error_%']

    Abs_Error  = |Eclipse_Dose - Model_Dose|
                 (Gy for D-type metrics, %pts for V-type metrics)
    Rel_Error_% = Abs_Error * 100 / Eclipse_Dose
    """
    rows     = []
    all_rois = sorted(set(list(eclipse_metrics) + list(ai_metrics)))

    for canonical in all_rois:
        if canonical.startswith("PTV"):
            metrics = PTV_METRICS
        elif canonical == "Penile_Bulb":
            metrics = PENILE_METRICS
        elif canonical in ("FemHead_L", "FemHead_R"):
            metrics = FEMORAL_METRICS
        else:
            metrics = OAR_METRICS

        for metric in metrics:
            ecl = eclipse_metrics.get(canonical, {}).get(metric)
            mdl = ai_metrics.get(canonical, {}).get(metric)

            # Absolute error (same units as the metric)
            if ecl is not None and mdl is not None:
                abs_err = round(abs(ecl - mdl), 2)
            else:
                abs_err = None

            # Relative error: Abs_Error * 100 / Eclipse_Dose (%)
            if abs_err is not None and ecl not in (None, 0.0):
                rel_err = round(abs_err * 100.0 / ecl, 2)
            else:
                rel_err = None

            rows.append({
                "ROI"          : canonical,
                "Metric"       : metric,
                "Eclipse_Dose" : ecl,
                "Model_Dose"   : mdl,
                "Abs_Error"    : abs_err,
                "Rel_Error_%"  : rel_err,
            })

    return pd.DataFrame(
        rows,
        columns=["ROI", "Metric", "Eclipse_Dose", "Model_Dose",
                 "Abs_Error", "Rel_Error_%"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eclipse vs AI-predicted dose DVH comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "patient_dir", type=str,
        help="Folder containing CT, RTSTRUCT, RTPLAN, and exactly 2 RTDOSE files",
    )
    parser.add_argument(
        "--output-dir", default=".", type=str,
        help="Directory to write the CSV output",
    )
    args = parser.parse_args()

    # ── 1. Identify files ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DVH COMPARISON — File Discovery")
    print("=" * 60)
    rtstruct_path, dose_eclipse, dose_ai, patient_id = \
        scan_dicom_dir(args.patient_dir)

    # ── 2. Calculate DVH metrics ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Eclipse DVH")
    print("=" * 60)
    eclipse_metrics = calculate_dvh_metrics(rtstruct_path, dose_eclipse)

    print("\n" + "=" * 60)
    print("  AI DVH")
    print("=" * 60)
    ai_metrics = calculate_dvh_metrics(rtstruct_path, dose_ai)

    # ── 3. Build and display DataFrame ────────────────────────────────────────
    df = build_dataframe(eclipse_metrics, ai_metrics)

    print("\n" + "=" * 60)
    print(f"  DVH COMPARISON — Patient: {patient_id}")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # ── 4. Save CSV ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{patient_id}_dvh_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[dvh] Saved → {out_csv}")

    return df


if __name__ == "__main__":
    main()
