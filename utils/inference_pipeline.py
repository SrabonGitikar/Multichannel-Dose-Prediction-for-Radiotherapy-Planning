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
from utils.create_dummy_plan import create_dummy_plan_dicom

# ─────────────────────────────────────────────────────────────────────────────
# Structure name patterns (same as dicom_to_nnunet.py)
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTVP$",
        r"^CTV_?60", r"^PTV_?60",
        r"^CTV_62", r"^PTV_62", r"^CTV62$", r"^PTV62$",
        r"^CTV_55", r"^PTV_55", r"^CTV55$", r"^PTV55$",
        r"^CTV_54", r"^PTV_54", r"^CTV54$", r"^PTV54$",
        r"^CTV_36", r"^PTV_36", r"^CTV 36", r"^PTV 36",
        r"^CTV_44", r"^PTV_44", r"^CTV_25", r"^PTV_25",
        r"^CTV 25", r"^PTV 25",
    ],
    "Bladder":     [r"^Bladder$", r"^BLADDER$"],
    "Anorectum":   [r"^Anorectum$", r"^ANORECTUM$", r"^Rectum$"],
    "Bag_Bowel":   [r"^Bag_?Bowel$", r"^Bag_?Bowel\s+NOS.*", r"^BagBowel.*"],
    "Body":        [r"^Body$", r"^EXTERNAL$", r"^Patient$", r"^Skin$"],
    "Femur_L":     [r"^Femur_?Head_?L.*", r"^L_?Femur.*", r"^Left_?Femur.*"],
    "Femur_R":     [r"^Femur_?Head_?R.*", r"^R_?Femur.*", r"^Right_?Femur.*"],
    "Penile_Bulb": [r"^Penile_?Bulb.*", r"^PenileBulb.*", r"^Penile Bulb.*"],
}

BODY_HU_THRESHOLD = -300.0
TARGET_SPACING    = (1.27, 1.27, 2.5)     # must match training
PATCH_SIZE        = (128, 128, 64)         # must match training
PRESCRIPTION_GY   = 75.0
CHANNELS          = ["0000", "0001", "0002", "0003", "0004", "0005", "0006"]
SAD_MM            = 1000.0
PENUMBRA_MM       = 7.0
DEFAULT_GANTRY_ANGLES = [0, 51, 102, 154, 205, 257, 308]


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

def _parse_gantry_angles_inf(plan_files):
    if not plan_files:
        print("  [pipeline] No RTPlan — defaulting to 7 equispaced gantry angles.")
        return DEFAULT_GANTRY_ANGLES
    try:
        plan = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
        angles = []
        if hasattr(plan, 'BeamSequence'):
            for beam in plan.BeamSequence:
                d_type = getattr(beam, 'TreatmentDeliveryType', '')
                b_type = getattr(beam, 'BeamType', '')
                if d_type == 'TREATMENT' or b_type == 'STATIC':
                    if hasattr(beam, 'ControlPointSequence') and len(beam.ControlPointSequence) > 0:
                        cp0 = beam.ControlPointSequence[0]
                        if hasattr(cp0, 'GantryAngle'):
                            angles.append(float(cp0.GantryAngle))
        if not angles:
            print("  [pipeline] No gantry angles — defaulting to 7 equispaced angles.")
            return DEFAULT_GANTRY_ANGLES
        unique_angles = sorted(set(angles))
        print(f"  [pipeline] Gantry Angles: {unique_angles}")
        return unique_angles
    except Exception as e:
        print(f"  [pipeline] RTPlan read error ({e}) — using fallback angles.")
        return DEFAULT_GANTRY_ANGLES


def _generate_bev_beam_mask(plan_files, ct_image, ptv_mask_array):
    """IEC 61217 BEV frustum mask — mirrors dicom_to_nnunet.py exactly."""
    gantry_angles = _parse_gantry_angles_inf(plan_files)
    shape_zyx = sitk.GetArrayViewFromImage(ct_image).shape
    origin    = np.array(ct_image.GetOrigin())
    spacing   = np.array(ct_image.GetSpacing())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    zi, yi, xi = np.meshgrid(np.arange(shape_zyx[0]), np.arange(shape_zyx[1]),
                              np.arange(shape_zyx[2]), indexing='ij')
    voxel_indices = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)
    phys_pts = origin + (voxel_indices * spacing) @ direction.T

    ptv_z, ptv_y, ptv_x = np.where(ptv_mask_array > 0.5)
    if len(ptv_z) == 0:
        print("  [pipeline] WARNING: PTV empty — blank beam mask.")
        return np.zeros(shape_zyx, dtype=np.float32)

    iso_idx  = np.array([ptv_x.mean(), ptv_y.mean(), ptv_z.mean()])
    iso_phys = origin + (iso_idx * spacing) @ direction.T
    print(f"  [pipeline] PTV isocenter (mm): {np.round(iso_phys, 1)}")

    beam_flat = np.zeros(len(phys_pts), dtype=np.float32)
    for angle_deg in gantry_angles:
        theta    = np.deg2rad(angle_deg)
        beam_dir = np.array([np.sin(theta), -np.cos(theta), 0.0])
        source   = iso_phys - beam_dir * SAD_MM
        vec      = phys_pts - source
        depth    = vec @ beam_dir
        perp_dist = np.linalg.norm(vec - depth[:, np.newaxis] * beam_dir, axis=1)

        ptv_indices = np.stack([ptv_x, ptv_y, ptv_z], axis=1)
        ptv_phys    = origin + (ptv_indices * spacing) @ direction.T
        vec_ptv     = ptv_phys - source
        depth_ptv   = vec_ptv @ beam_dir
        perp_ptv    = np.linalg.norm(vec_ptv - depth_ptv[:, np.newaxis] * beam_dir, axis=1)

        half_angle = (np.arctan2(perp_ptv.max(), depth_ptv[depth_ptv > 0].min())
                      if depth_ptv.max() > 0 else np.deg2rad(5.0))
        radius = np.abs(depth) * np.tan(half_angle) + PENUMBRA_MM
        beam_flat[(depth > 0) & (perp_dist <= radius)] = 1.0

    beam_zyx = beam_flat.reshape(shape_zyx)
    print(f"  [pipeline] BEV voxels: {int(beam_zyx.sum()):,}")
    return beam_zyx



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
    Writes 7 NIfTI channel files to *images_dir* matching the training layout exactly.
    Ch0=CT, Ch1=PTV_binary, Ch2=Bladder_SDM, Ch3=Anorectum_SDM,
    Ch4=Body_Mask, Ch5=Penile_Bulb, Ch6=BEV_Beam_Frustum
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

    # Core masks
    ptv_mask     = _union_contours_to_mask(rs_ds, ptv_names, ct_image)
    bladder_mask = _contour_to_mask(rs_ds, bladder_name, ct_image)
    anorect_mask = _contour_to_mask(rs_ds, anorect_name, ct_image)
    assert ptv_mask.sum() > 0, "PTV mask is empty — check contour names."

    # Signed distance maps
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = _signed_distance_map(bladder_mask, spacing_zyx)
    anorect_sdm = _signed_distance_map(anorect_mask, spacing_zyx)

    # Ch4: Body Mask — prefer RTStruct contour, fall back to HU threshold
    body_name = _match_structure(roi_names, "Body")
    if body_name:
        print(f"  [preprocess] Body contour: {body_name}")
        body_mask = _contour_to_mask(rs_ds, body_name, ct_image).astype(np.float32)
    else:
        print("  [preprocess] No Body contour — using HU threshold fallback.")
        body_mask = (ct_array > BODY_HU_THRESHOLD).astype(np.float32)

    # Ch5: Penile Bulb — empty mask if absent
    penile_name = _match_structure(roi_names, "Penile_Bulb")
    if penile_name:
        penile_mask = _contour_to_mask(rs_ds, penile_name, ct_image).astype(np.float32)
        print(f"  [preprocess] Penile_Bulb: {penile_name}")
    else:
        penile_mask = np.zeros_like(ptv_mask, dtype=np.float32)
        print("  [preprocess] Penile_Bulb: NOT FOUND — using empty mask.")

    # Ch6: BEV Frustum Beam Mask
    beam_mask = _generate_bev_beam_mask(plan_files, ct_image, ptv_mask)

    # Write channels in training order
    def _save(arr, suffix):
        path = os.path.join(images_dir, f"{case_name}_{suffix}.nii.gz")
        sitk.WriteImage(_numpy_to_sitk(arr.astype(np.float32), ct_image), path)
        return path

    _save(ct_array,    "0000")   # ch_0: CT
    _save(ptv_mask,    "0001")   # ch_1: PTV binary mask (CreateDiscretePTVMapd consumes this)
    _save(bladder_sdm, "0002")   # ch_2: Bladder SDM
    _save(anorect_sdm, "0003")   # ch_3: Anorectum SDM
    _save(body_mask,   "0004")   # ch_4: Body Mask
    _save(penile_mask, "0005")   # ch_5: Penile Bulb
    _save(beam_mask,   "0006")   # ch_6: BEV Frustum

    print(f"  [preprocess] 7 channels saved → {images_dir}/{case_name}_000[0-6].nii.gz")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(images_dir: str, case_name: str, model_path: str,
                  output_dir: str) -> str:
    """
    Run sliding-window inference and save predicted dose as NIfTI.
    Channel layout fed to model (must exactly match training):
      0=CT(norm)  1=discrete_ptv  2=Bladder_SDM  3=Anorectum_SDM
      4=Body  5=PenileBulb  6=BEV_Beam
    """
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        NormalizeIntensityd, ConcatItemsd, ToTensord, DeleteItemsd,
        MapTransform,
    )
    from monai.data import Dataset, DataLoader

    # ---- Replicate CreateDiscretePTVMapd from training script ----------------
    SIB_ORDER = [
        ("PTV25", 25.0), ("PTV36", 36.0), ("PTV44", 44.0),
        ("PTV54", 54.0), ("PTV55", 55.0), ("PTV60", 60.0),
    ]
    SIB_CANONICAL = {
        r"^ptv.*60|^ctv.*60|^ctvp$": "PTV60",
        r"^ptv.*62|^ctv.*62": "PTV60",
        r"^ptv.*55|^ctv.*55": "PTV55",
        r"^ptv.*54|^ctv.*54": "PTV54",
        r"^ptv.*44|^ctv.*44": "PTV44",
        r"^ptv.*36|^ctv.*36": "PTV36",
        r"^ptv.*25|^ctv.*25": "PTV25",
    }

    class _CreateDiscretePTVMapd(MapTransform):
        """Mirrors CreateDiscretePTVMapd from train_dummy_physics_new.py."""
        def __call__(self, data):
            d = dict(data)
            discrete_ptv = torch.zeros_like(d["ch_0"])
            # Build individual PTV keys from ch_1 using canonical names
            # For inference we only have the union mask in ch_1;
            # assign it the highest dose level present (PTV60 → 60.0)
            ptv_union = d["ch_1"]
            discrete_ptv = torch.where(
                ptv_union >= 0.5,
                torch.tensor(60.0, device=ptv_union.device),
                discrete_ptv,
            )
            d["discrete_ptv"] = discrete_ptv
            return d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [inference] Device: {device}")

    model = UNet(
        spatial_dims=3, in_channels=7, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    ).to(device)

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  [inference] Model loaded: {model_path}")

    _CH_KEYS  = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"]
    _CH_MODES = ("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest", "nearest")

    transforms = Compose([
        LoadImaged(keys=_CH_KEYS),
        EnsureChannelFirstd(keys=_CH_KEYS),
        Spacingd(keys=_CH_KEYS, pixdim=TARGET_SPACING, mode=_CH_MODES),
        _CreateDiscretePTVMapd(keys=["ch_0"]),   # ch_1 → discrete_ptv
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        # Concat order mirrors ConcatItemsd in training script exactly
        ConcatItemsd(keys=["ch_0", "discrete_ptv", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"],
                     name="image"),
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
    # Ch_4 = Body Mask (correct index after training layout fix)
    body_hard = (inputs[:, 4:5, ...] > 0.5).float()
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

    # ── 0. Ensure RTPLAN exists (create dummy if missing) ────────────────────
    print("[Step 0/3] Checking for RTPLAN...")
    _has_rtplan = False
    for _f in Path(dicom_dir).rglob("*.dcm"):
        try:
            _ds = pydicom.dcmread(str(_f), stop_before_pixels=True, force=True)
            if getattr(_ds, "Modality", "") == "RTPLAN":
                _has_rtplan = True
                print(f"  RTPLAN found: {_f.name}")
                break
        except Exception:
            continue

    if not _has_rtplan:
        print("  No RTPLAN found — generating dummy plan...")
        create_dummy_plan_dicom(dicom_dir)

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
