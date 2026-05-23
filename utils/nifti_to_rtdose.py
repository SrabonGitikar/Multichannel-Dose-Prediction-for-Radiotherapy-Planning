"""
utils/nifti_to_rtdose.py

Convert a predicted dose NIfTI file into a valid RTDOSE DICOM, using
patient/study/geometry metadata extracted directly from a folder that
contains CT + RTSTRUCT DICOMs.

Usage
-----
    from utils.nifti_to_rtdose import nifti_to_rtdose_dicom

    out = nifti_to_rtdose_dicom(
        ct_rs_dir   = "/data/patient_001/dicom",
        nifti_path  = "/data/output/patient_001_predicted_dose.nii.gz",
        output_dir  = "/data/output",
    )
    print("Saved:", out)

CLI
---
    python utils/nifti_to_rtdose.py \\
        --ct-rs-dir  /data/patient_001/dicom \\
        --nifti-path /data/output/patient_001_predicted_dose.nii.gz \\
        --output-dir /data/output
"""

import os
import datetime
import argparse
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scan_dicom_folder(folder: str):
    """
    Recursively scan *folder* and return separate lists:
        ct_slices  – pydicom datasets for all CT slices (headers only)
        rtstruct   – first RTSTRUCT dataset found (headers only), or None
    Raises AssertionError if no CT found or no RTSTRUCT found.
    """
    ct_slices = []
    rtstruct  = None

    for f in sorted(Path(folder).rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = getattr(ds, "Modality", "")
            if modality == "CT":
                ct_slices.append(ds)
            elif modality == "RTSTRUCT" and rtstruct is None:
                rtstruct = ds
        except Exception:
            continue

    assert ct_slices, f"No CT DICOM slices found in: {folder}"
    assert rtstruct  is not None, (
        f"No RTSTRUCT DICOM found in: {folder}\n"
        "The folder must contain at least one file with Modality == 'RTSTRUCT'."
    )

    ct_slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return ct_slices, rtstruct


def _build_sitk_reference(ct_slices, z_positions, spacing_xy, spacing_z, iop, origin):
    """Build an empty SimpleITK image that matches the CT volume geometry."""
    row_cos = np.array(iop[0:3])
    col_cos = np.array(iop[3:6])
    nor_cos = np.cross(row_cos, col_cos)
    direction = tuple(row_cos.tolist() + col_cos.tolist() + nor_cos.tolist())

    ct_ref_img = sitk.Image(
        int(ct_slices[0].Columns),
        int(ct_slices[0].Rows),
        len(ct_slices),
        sitk.sitkFloat32,
    )
    ct_ref_img.SetSpacing((spacing_xy[1], spacing_xy[0], spacing_z))
    ct_ref_img.SetOrigin(origin)
    ct_ref_img.SetDirection(direction)
    return ct_ref_img


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def nifti_to_rtdose_dicom(
    ct_rs_dir:  str,
    nifti_path: str,
) -> str:
    """
    Parameters
    ----------
    ct_rs_dir  : folder that contains all CT slices AND the RTSTRUCT .dcm
                 The output RTDOSE .dcm is saved into this same folder.
    nifti_path : predicted dose NIfTI (.nii / .nii.gz), values in Gy

    Returns
    -------
    str : absolute path to the saved RTDOSE DICOM
    """

    # ── 1. Scan folder for CT + RTSTRUCT ─────────────────────────────────────
    ct_slices, rs_ds = _scan_dicom_folder(ct_rs_dir)
    ct_ref = ct_slices[0]

    z_positions = [float(s.ImagePositionPatient[2]) for s in ct_slices]
    spacing_xy  = [float(v) for v in ct_ref.PixelSpacing]   # [row_sp, col_sp]
    spacing_z   = (
        abs(z_positions[1] - z_positions[0])
        if len(z_positions) > 1
        else float(getattr(ct_ref, "SliceThickness", 2.5))
    )
    origin   = [float(v) for v in ct_ref.ImagePositionPatient]
    iop      = [float(v) for v in ct_ref.ImageOrientationPatient]
    ct_rows  = int(ct_ref.Rows)
    ct_cols  = int(ct_ref.Columns)
    n_slices = len(ct_slices)

    patient_id = getattr(ct_ref, "PatientID", "unknown")
    print(f"[nifti_to_rtdose] CT: {ct_cols}×{ct_rows}×{n_slices}  "
          f"spacing=({spacing_xy[1]:.3f},{spacing_xy[0]:.3f},{spacing_z:.3f}) mm")
    print(f"[nifti_to_rtdose] Patient: {getattr(ct_ref,'PatientName','')} | ID: {patient_id}")
    print(f"[nifti_to_rtdose] RTSTRUCT: {rs_ds.SOPInstanceUID}")

    # ── 2. Resample NIfTI onto CT grid ───────────────────────────────────────
    ct_ref_img = _build_sitk_reference(ct_slices, z_positions, spacing_xy, spacing_z, iop, origin)
    pred_sitk  = sitk.ReadImage(nifti_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    pred_on_ct = resampler.Execute(pred_sitk)

    pred_gy = np.clip(
        sitk.GetArrayFromImage(pred_on_ct).astype(np.float32), 0.0, None
    )  # (Z, Y, X) = (frames, rows, cols)
    print(f"[nifti_to_rtdose] Dose resampled: {pred_gy.shape}  "
          f"[{pred_gy.min():.2f}, {pred_gy.max():.2f}] Gy")

    # ── 3. Gy → uint32 pixel values ──────────────────────────────────────────
    max_dose    = float(pred_gy.max()) or 1.0
    new_scaling = max_dose / (2**32 - 1)
    pixels      = np.round(pred_gy / new_scaling).clip(0, 2**32 - 1).astype(np.uint32)

    # ── 4. Build RTDOSE DICOM ─────────────────────────────────────────────────
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")
    sop_uid  = generate_uid()

    timestamp    = now.strftime("%Y%m%d_%H%M%S")
    out_filename = f"predicted-dose-{timestamp}.dcm"
    out_path     = os.path.join(os.path.abspath(ct_rs_dir), out_filename)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.481.2"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID     = generate_uid()
    file_meta.ImplementationVersionName  = "AI_DOSE_1.0"

    rtdose = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    rtdose.is_implicit_VR   = False
    rtdose.is_little_endian = True

    # Mandatory identification tags
    rtdose.SpecificCharacterSet = "ISO_IR 192"
    rtdose.SOPClassUID          = "1.2.840.10008.5.1.4.1.1.481.2"
    rtdose.SOPInstanceUID       = sop_uid
    rtdose.Modality             = "RTDOSE"
    rtdose.InstanceCreationDate = date_str
    rtdose.InstanceCreationTime = time_str
    rtdose.ContentDate          = date_str
    rtdose.ContentTime          = time_str

    # Patient — copied from CT
    for tag in ["PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                "PatientAge", "PatientWeight"]:
        if hasattr(ct_ref, tag):
            setattr(rtdose, tag, getattr(ct_ref, tag))

    # Study — MUST match CT so Slicer links them in the same study
    for tag in ["StudyInstanceUID", "StudyDate", "StudyTime", "StudyID",
                "StudyDescription", "AccessionNumber", "ReferringPhysicianName"]:
        if hasattr(ct_ref, tag):
            setattr(rtdose, tag, getattr(ct_ref, tag))

    # Frame of Reference — MUST match CT for spatial overlay in Slicer
    for tag in ["FrameOfReferenceUID", "PositionReferenceIndicator"]:
        if hasattr(ct_ref, tag):
            setattr(rtdose, tag, getattr(ct_ref, tag))

    # Series — new series within the same study
    rtdose.SeriesInstanceUID    = generate_uid()
    rtdose.SeriesNumber         = "900"
    rtdose.SeriesDescription    = "Predicted Dose (AI)"
    rtdose.SeriesDate           = date_str
    rtdose.SeriesTime           = time_str
    rtdose.Manufacturer         = "AI Dose Prediction"
    rtdose.ManufacturerModelName = "DoseNet-v1"
    rtdose.SoftwareVersions     = "1.0"

    # Image geometry
    rtdose.Rows                    = ct_rows
    rtdose.Columns                 = ct_cols
    rtdose.NumberOfFrames          = n_slices
    rtdose.PixelSpacing            = [f"{spacing_xy[0]:.6f}", f"{spacing_xy[1]:.6f}"]
    rtdose.SliceThickness          = f"{spacing_z:.6f}"
    rtdose.ImagePositionPatient    = [f"{v:.6f}" for v in origin]
    rtdose.ImageOrientationPatient = [f"{v:.6f}" for v in iop]
    rtdose.GridFrameOffsetVector   = [f"{z - z_positions[0]:.4f}" for z in z_positions]
    rtdose.FrameIncrementPointer   = pydicom.dataelem.Tag(0x3004, 0x000C)

    # Pixel data
    rtdose.SamplesPerPixel          = 1
    rtdose.PhotometricInterpretation = "MONOCHROME2"
    rtdose.BitsAllocated            = 32
    rtdose.BitsStored               = 32
    rtdose.HighBit                  = 31
    rtdose.PixelRepresentation      = 0
    rtdose.PixelData                = pixels.tobytes()

    # RT Dose specific
    rtdose.DoseUnits                     = "GY"
    rtdose.DoseType                      = "PHYSICAL"
    rtdose.DoseSummationType             = "PLAN"
    rtdose.DoseGridScaling               = f"{new_scaling:.10e}"
    rtdose.TissueHeterogeneityCorrection = ["IMAGE"]

    # Link RTSTRUCT
    ref_item = Dataset()
    ref_item.ReferencedSOPClassUID    = rs_ds.SOPClassUID
    ref_item.ReferencedSOPInstanceUID = rs_ds.SOPInstanceUID
    rtdose.ReferencedStructureSetSequence = Sequence([ref_item])

    pydicom.dcmwrite(out_path, rtdose)
    print(f"[nifti_to_rtdose] Saved: {out_path}")
    print(f"[nifti_to_rtdose] FrameOfReferenceUID : {rtdose.FrameOfReferenceUID}")
    print(f"[nifti_to_rtdose] StudyInstanceUID    : {rtdose.StudyInstanceUID}")
    print(f"[nifti_to_rtdose] DoseGridScaling     : {new_scaling:.6e} Gy/count")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert predicted dose NIfTI → RTDOSE DICOM"
    )
    parser.add_argument("--ct-rs-dir",  required=True,
                        help="Folder containing CT slices + RTSTRUCT .dcm")
    parser.add_argument("--nifti-path", required=True,
                        help="Predicted dose NIfTI (.nii / .nii.gz), values in Gy")
    args = parser.parse_args()

    saved = nifti_to_rtdose_dicom(
        ct_rs_dir  = args.ct_rs_dir,
        nifti_path = args.nifti_path,
    )
    print(f"Done: {saved}")
