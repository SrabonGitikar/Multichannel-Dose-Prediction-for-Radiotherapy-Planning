"""
inference_pipeline.py  —  End-to-end dose prediction pipeline (entry point)
============================================================================
Given a DICOM folder (CT + RTSTRUCT + optional RTPLAN), this script:

  Step 1 — Preprocess   : DICOM  →  NIfTI input files (driven by config.yml)
  Step 2 — Inference    : NIfTI  →  predicted dose NIfTI  (at target_spacing)
  Step 3 — RTDOSE build : NIfTI  →  RTDOSE DICOM (resampled to native CT grid)
  Step 4 — Cleanup      : remove temp dir (unless --keep-temp)

All logic lives in utils/inference_pipeline.py which mirrors the val_transforms
from "01 ICON/utils/training.py" exactly.  This file is a thin CLI wrapper.

Usage (CLI)
-----------
    python inference_pipeline.py \\
        --dicom-dir  "/data/patient_001/dicom"    \\
        --config     "01 ICON/config/config.yml"  \\
        --model      best_dose_model_XXX.pth      \\
        --dose-spacing 2.5                        \\
        --keep-temp
"""

import argparse
import sys
from pathlib import Path

# ── Make sure the project root is on sys.path so utils/ resolves ─────────────
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Delegate everything to the self-contained utils pipeline ─────────────────
from utils.inference_pipeline import run_pipeline   # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end AI dose prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dicom-dir", required=True, type=str,
        help="Folder containing CT, RTSTRUCT, and (optionally) RTPLAN DICOMs",
    )
    parser.add_argument(
        "--config", default="01 ICON/config/config.yml", type=str,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--model", required=True, type=str,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--dose-spacing", default=None, type=float,
        help="Z-spacing (mm) for the output RTDOSE DICOM; defaults to config value",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Retain the temporary NIfTI workspace directory after completion",
    )
    args = parser.parse_args()

    rtdose_path = run_pipeline(
        dicom_dir      = args.dicom_dir,
        config_path    = args.config,
        model_path     = args.model,
        dose_spacing_mm= args.dose_spacing,
        keep_temp      = args.keep_temp,
    )

    if rtdose_path:
        print(f"\n✓  RTDOSE written to: {rtdose_path}")
    else:
        print("\n⚠  Pipeline finished but RTDOSE was not generated.")
        print("   Check that utils/nifti_to_rtdose.py is on the Python path.")
