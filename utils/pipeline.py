"""
utils/pipeline.py  —  End-to-end dose prediction pipeline
===========================================================
Given a DICOM folder (CT + RTSTRUCT + optional RTPLAN), this script:

  Step 1 — Preprocess   : DICOM  →  6-channel NIfTI input files  (temp dir)
  Step 2 — Inference    : NIfTI  →  predicted dose NIfTI          (temp dir)
  Step 3 — RTDOSE build : NIfTI  →  RTDOSE DICOM                  (ct_rs_dir)
  Step 4 — Cleanup      : remove temp dir (unless --keep-temp)

Usage (CLI)
-----------
    python utils/pipeline.py \\
        --dicom-dir  "/data/patient_001/dicom" \\
        --model      best_dose_model_physics_L1.pth \\
        --dose-spacing 2.5

Usage (Python)
--------------
    from utils.pipeline import run_pipeline

    out = run_pipeline(
        dicom_dir       = "/data/patient_001/dicom",
        model_path      = "best_dose_model_physics_L1.pth",
        dose_spacing_mm = 2.5,
    )
    print("RTDOSE saved:", out)
"""

import os
import re
import sys
import shutil
import argparse
import tempfile

import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon

# ── project root on sys.path so sibling modules resolve cleanly ──────────────
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.nifti_to_rtdose import nifti_to_rtdose_dicom

# ─────────────────────────────────────────────────────────────────────────────
# Structure name patterns (same as dicom_to_nnunet.py)
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTVP$", r"^CTV_62", r"^PTV_62", r"^CTV62$", r"^PTV62$",
        r"^CTV_36", r"^PTV_36", r"^CTV 36", r"^PTV 36",
        r"^CTV_44", r"^PTV_44", r"^CTV_25", r"^PTV_25",
        r"^CTV 25", r"^PTV 25",
    ],
    "Bladder":   [r"^Bladder$", r"^BLADDER$"],
    "Anorectum": [r"^Anorectum$", r"^ANORECTUM$", r"^Rectum$"],
    "Bag_Bowel": [r"^Bag_?Bowel$", r"^Bag_?Bowel\s+NOS.*", r"^BagBowel.*"],
    "Femur_L":   [r"^Femur_?Head_?L.*", r"^L_?Femur.*", r"^Left_?Femur.*"],
    "Femur_R":   [r"^Femur_?Head_?R.*", r"^R_?Femur.*", r"^Right_?Femur.*"],
}

BODY_HU_THRESHOLD = -300.0
TARGET_SPACING    = (1.27, 1.27, 2.5)     # must match training
PATCH_SIZE        = (128, 128, 64)         # must match training
PRESCRIPTION_GY   = 75.0
CHANNELS          = ["0000", "0001", "0002", "0003", "0004", "0005"]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers  (adapted from dicom_to_nnunet.py — no RTDose needed)
# ─────────────────────────────────────────────────────────────────────────────

def _match_structure(roi_names, structure_type):
    for pattern in STRUCTURE_PATTERNS[structure_type]:
        for name in roi_names:
            if re.match(pattern, name, re.IGNORECASE):
                return name
    return None

def _match_all_structures(roi_names, structure_type):
    matched = []
    for name in roi_names:
        for pattern in STRUCTURE_PATTERNS[structure_type]:
            if re.match(pattern, name, re.IGNORECASE):
                matched.append(name)
                break
    return matched

def _contour_to_mask(rs_ds, roi_name, ct_image):
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask = np.zeros(ct_array.shape, dtype=np.uint8)
    roi_number = None
    for roi in rs_ds.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber
            break
    if roi_number is None:
        return mask
    contour_seq = None
    for rc in rs_ds.ROIContourSequence:
        if rc.ReferencedROINumber == roi_number:
            contour_seq = rc
            break
    if contour_seq is None or not hasattr(contour_seq, "ContourSequence"):
        return mask
    for contour in contour_seq.ContourSequence:
        pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        pixel_coords = [
            ct_image.TransformPhysicalPointToContinuousIndex(
                (float(p[0]), float(p[1]), float(p[2]))
            )
            for p in pts
        ]
        pixel_coords = np.array(pixel_coords)
        slice_idx = int(round(pixel_coords[0, 2]))
        if 0 <= slice_idx < mask.shape[0]:
            rr, cc = polygon(pixel_coords[:, 1], pixel_coords[:, 0],
                             shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1
    return mask

def _union_contours_to_mask(rs_ds, roi_names_list, ct_image):
    ct_array = sitk.GetArrayFromImage(ct_image)
    union = np.zeros(ct_array.shape, dtype=np.uint8)
    for name in roi_names_list:
        union = np.maximum(union, _contour_to_mask(rs_ds, name, ct_image))
    return union

def _signed_distance_map(binary_mask, spacing_mm):
    if binary_mask.sum() == 0:
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0
    d_out = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)
    d_in  = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)
    return (d_out - d_in).astype(np.float32)

def _generate_beam_mask(plan_files, ct_image, ptv_mask):
    if not plan_files:
        print("  [pipeline] WARNING: No RTPLAN found — beam prior set to zeros.")
        return np.zeros(sitk.GetArrayViewFromImage(ct_image).shape, dtype=np.float32)

    plan = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
    isocenter_mm, gantry_angles = None, []
    if hasattr(plan, "BeamSequence"):
        for beam in plan.BeamSequence:
            b_type = getattr(beam, "BeamType", "")
            d_type = getattr(beam, "TreatmentDeliveryType", "")
            if b_type == "STATIC" or d_type == "TREATMENT":
                if hasattr(beam, "ControlPointSequence") and beam.ControlPointSequence:
                    cp0 = beam.ControlPointSequence[0]
                    if isocenter_mm is None and hasattr(cp0, "IsocenterPosition"):
                        isocenter_mm = np.array(cp0.IsocenterPosition)
                    if hasattr(cp0, "GantryAngle"):
                        gantry_angles.append(float(cp0.GantryAngle))

    if isocenter_mm is None or not gantry_angles:
        print("  [pipeline] WARNING: No isocenter/gantry angles — beam prior set to zeros.")
        return np.zeros(sitk.GetArrayViewFromImage(ct_image).shape, dtype=np.float32)

    ptv_z, ptv_y, ptv_x = np.where(ptv_mask > 0.5)
    if len(ptv_x):
        pts = np.array([
            ct_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            for x, y, z in zip(ptv_x, ptv_y, ptv_z)
        ])
        cylinder_radius_mm = np.max(np.linalg.norm(pts - isocenter_mm, axis=1)) + 10.0
    else:
        cylinder_radius_mm = 50.0

    shape_zyx = sitk.GetArrayViewFromImage(ct_image).shape
    shape_xyz = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
    spacing   = np.array(ct_image.GetSpacing())
    origin    = np.array(ct_image.GetOrigin())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    xi, yi, zi = np.meshgrid(
        np.arange(shape_xyz[0]), np.arange(shape_xyz[1]), np.arange(shape_xyz[2]),
        indexing="ij"
    )
    indices  = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)
    physical = origin + np.dot(indices * spacing, direction.T)
    beam_flat = np.zeros(len(physical), dtype=np.float32)

    for angle in gantry_angles:
        theta = np.deg2rad(angle)
        bdir  = np.array([np.sin(theta), -np.cos(theta), 0.0])
        bdir /= np.linalg.norm(bdir)
        vec   = physical - isocenter_mm
        proj  = np.sum(vec * bdir, axis=1, keepdims=True) * bdir
        perp  = np.linalg.norm(vec - proj, axis=1)
        beam_flat[perp <= cylinder_radius_mm] = 1.0

    beam_xyz = beam_flat.reshape(shape_xyz)
    return np.transpose(beam_xyz, (2, 1, 0))  # ZYX


def _numpy_to_sitk(array, reference):
    img = sitk.GetImageFromArray(array)
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    img.SetDirection(reference.GetDirection())
    return img


def _scan_dicom_dir(dicom_dir):
    """Return (ct_image, rs_ds, plan_files) from a DICOM folder."""
    dicom_dir = str(dicom_dir)

    # ── CT via GDCM series reader ────────────────────────────────────────────
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    assert series_ids, f"No DICOM series found in: {dicom_dir}"
    best, best_n = None, 0
    for sid in series_ids:
        fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(fnames) > best_n:
            best_n, best = len(fnames), fnames
    reader.SetFileNames(best)
    ct_image = reader.Execute()
    print(f"  [preprocess] CT loaded: shape={sitk.GetArrayFromImage(ct_image).shape}  "
          f"spacing={ct_image.GetSpacing()}")

    # ── RTSTRUCT + RTPLAN via Modality scan ──────────────────────────────────
    rs_ds = None
    plan_files = []
    for root, _, files in os.walk(dicom_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                mod = getattr(ds, "Modality", "")
                if mod == "RTSTRUCT" and rs_ds is None:
                    rs_ds = pydicom.dcmread(fpath)
                    print(f"  [preprocess] RTSTRUCT: {fpath}")
                elif mod == "RTPLAN":
                    plan_files.append(fpath)
            except Exception:
                continue

    assert rs_ds is not None, (
        f"ERROR: No RTSTRUCT found in {dicom_dir}.\n"
        "Place the RTSTRUCT .dcm in the same folder tree."
    )
    return ct_image, rs_ds, plan_files


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Preprocess DICOM → 6-channel NIfTI
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dicom(dicom_dir: str, images_dir: str, case_name: str = "patient_001") -> None:
    """
    Preprocess a DICOM folder for inference.
    Writes 6 NIfTI channel files to *images_dir* (RTDose is NOT required).
    """
    os.makedirs(images_dir, exist_ok=True)
    ct_image, rs_ds, plan_files = _scan_dicom_dir(dicom_dir)
    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing  = ct_image.GetSpacing()

    roi_names    = [roi.ROIName for roi in rs_ds.StructureSetROISequence]
    ptv_names    = _match_all_structures(roi_names, "PTV")
    bladder_name = _match_structure(roi_names, "Bladder")
    anorect_name = _match_structure(roi_names, "Anorectum")

    assert ptv_names,    "Cannot match any PTV/CTV structure in RTSTRUCT."
    assert bladder_name, "Cannot match Bladder structure in RTSTRUCT."
    assert anorect_name, "Cannot match Anorectum structure in RTSTRUCT."

    print(f"  [preprocess] PTV structures: {ptv_names}")
    print(f"  [preprocess] Bladder={bladder_name}  Anorectum={anorect_name}")

    # Masks
    ptv_mask      = _union_contours_to_mask(rs_ds, ptv_names, ct_image)
    bladder_mask  = _contour_to_mask(rs_ds, bladder_name, ct_image)
    anorect_mask  = _contour_to_mask(rs_ds, anorect_name, ct_image)
    assert ptv_mask.sum() > 0, "PTV mask is empty — check contour names."

    # Signed distance maps
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = _signed_distance_map(bladder_mask, spacing_zyx)
    anorect_sdm = _signed_distance_map(anorect_mask, spacing_zyx)

    # Beam prior
    beam_mask = _generate_beam_mask(plan_files, ct_image, ptv_mask)

    # Body mask
    body_mask = (ct_array > BODY_HU_THRESHOLD).astype(np.float32)

    # Write channels
    def _save(arr, suffix):
        path = os.path.join(images_dir, f"{case_name}_{suffix}.nii.gz")
        sitk.WriteImage(_numpy_to_sitk(arr.astype(np.float32), ct_image), path)
        return path

    _save(ct_array,      "0000")   # ch_0: CT
    _save(ptv_mask,      "0001")   # ch_1: PTV binary mask
    _save(bladder_sdm,   "0002")   # ch_2: Bladder SDM
    _save(anorect_sdm,   "0003")   # ch_3: Anorectum SDM
    _save(beam_mask,     "0004")   # ch_4: IMRT beam prior
    _save(body_mask,     "0005")   # ch_5: body mask

    print(f"  [preprocess] 6 channels saved → {images_dir}/{case_name}_000[0-5].nii.gz")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(images_dir: str, case_name: str, model_path: str,
                  output_dir: str) -> str:
    """
    Run sliding-window inference and save predicted dose as NIfTI.
    Returns the path to the saved .nii.gz file.
    """
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        NormalizeIntensityd, ConcatItemsd, ToTensord, DeleteItemsd,
    )
    from monai.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [inference] Device: {device}")

    model = UNet(
        spatial_dims=3, in_channels=6, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    ).to(device)

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  [inference] Model loaded: {model_path}")

    _CH_KEYS  = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5"]
    _CH_MODES = ("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest")

    transforms = Compose([
        LoadImaged(keys=_CH_KEYS),
        EnsureChannelFirstd(keys=_CH_KEYS),
        Spacingd(keys=_CH_KEYS, pixdim=TARGET_SPACING, mode=_CH_MODES),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=_CH_KEYS, name="image"),
        DeleteItemsd(keys=_CH_KEYS),
        ToTensord(keys=["image"]),
    ])

    pt_dict = {
        f"ch_{i}": os.path.join(images_dir, f"{case_name}_{ch}.nii.gz")
        for i, ch in enumerate(CHANNELS)
    }
    for path in pt_dict.values():
        assert os.path.exists(path), f"Missing channel file: {path}"

    ds     = Dataset(data=[pt_dict], transform=transforms)
    loader = DataLoader(ds, batch_size=1)
    batch  = next(iter(loader))
    inputs = batch["image"].to(device)

    ref_affine = nib.load(pt_dict["ch_0"]).affine

    print(f"  [inference] Input shape: {inputs.shape}")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = sliding_window_inference(
                inputs=inputs, roi_size=PATCH_SIZE,
                sw_batch_size=2, predictor=model,
                overlap=0.5, mode="gaussian",
            )

    outputs_activated = F.softplus(outputs.float())
    body_hard = (inputs[:, 5:6, ...] > 0.5).float()
    outputs_activated = outputs_activated * body_hard

    pred_dose = outputs_activated[0, 0].cpu().numpy() * PRESCRIPTION_GY
    pred_dose = np.clip(pred_dose, 0.0, None)

    print(f"  [inference] Dose range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")

    os.makedirs(output_dir, exist_ok=True)
    nifti_path = os.path.join(output_dir, f"{case_name}_predicted_dose.nii.gz")
    nib.save(nib.Nifti1Image(pred_dose.astype(np.float32), affine=ref_affine), nifti_path)
    print(f"  [inference] Dose NIfTI saved: {nifti_path}")
    return nifti_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    dicom_dir:       str,
    model_path:      str  = "best_dose_model_physics_L1.pth",
    dose_spacing_mm: float = 2.5,
    case_name:       str  = "patient_001",
    keep_temp:       bool = False,
) -> str:
    """
    Full end-to-end pipeline: DICOM folder → RTDOSE DICOM.

    Parameters
    ----------
    dicom_dir       : folder containing CT + RTSTRUCT (+ optional RTPLAN) DICOMs
    model_path      : path to trained model .pth checkpoint
    dose_spacing_mm : output RTDOSE voxel size in mm (default 2.5 mm → ~15 MB)
    case_name       : internal case identifier (default: patient_001)
    keep_temp       : if True, the temp NIfTI folder is not deleted after use

    Returns
    -------
    str : absolute path to the saved RTDOSE DICOM
    """
    dicom_dir = str(Path(dicom_dir).resolve())
    print(f"\n{'='*60}")
    print(f"  Pipeline start: {dicom_dir}")
    print(f"{'='*60}\n")

    # ── temp directory ────────────────────────────────────────────────────────
    tmp_root   = tempfile.mkdtemp(prefix="dose_pipeline_")
    images_dir = os.path.join(tmp_root, "imagesTr")
    pred_dir   = os.path.join(tmp_root, "predictions")

    try:
        # ── Step 1: Preprocess ────────────────────────────────────────────────
        print("[Step 1/3] Preprocessing DICOM → NIfTI channels...")
        preprocess_dicom(dicom_dir, images_dir, case_name)

        # ── Step 2: Inference ─────────────────────────────────────────────────
        print("\n[Step 2/3] Running model inference...")
        nifti_path = run_inference(images_dir, case_name, model_path, pred_dir)

        # ── Step 3: RTDOSE DICOM ──────────────────────────────────────────────
        print("\n[Step 3/3] Building RTDOSE DICOM...")
        rtdose_path = nifti_to_rtdose_dicom(
            ct_rs_dir       = dicom_dir,
            nifti_path      = nifti_path,
            dose_spacing_mm = dose_spacing_mm,
        )

    finally:
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"\n  Temp folder removed: {tmp_root}")
        else:
            print(f"\n  Temp folder kept at: {tmp_root}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  RTDOSE saved: {rtdose_path}")
    print(f"{'='*60}\n")
    return rtdose_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end dose prediction: DICOM → RTDOSE DICOM"
    )
    parser.add_argument("--dicom-dir", required=True,
                        help="Folder containing CT + RTSTRUCT (+ optional RTPLAN)")
    parser.add_argument("--model", default="best_dose_model_physics_L1.pth",
                        help="Path to trained model .pth (default: best_dose_model_physics_L1.pth)")
    parser.add_argument("--dose-spacing", type=float, default=2.5,
                        help="Output dose grid spacing in mm (default: 2.5)")
    parser.add_argument("--case-name", default="patient_001",
                        help="Internal case identifier (default: patient_001)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep intermediate NIfTI files after completion")
    args = parser.parse_args()

    rtdose = run_pipeline(
        dicom_dir       = args.dicom_dir,
        model_path      = args.model,
        dose_spacing_mm = args.dose_spacing,
        case_name       = args.case_name,
        keep_temp       = args.keep_temp,
    )
    print(f"Done: {rtdose}")
