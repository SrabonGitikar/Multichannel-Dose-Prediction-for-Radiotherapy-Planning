"""
utils/inference_pipeline.py -- End-to-end dose prediction pipeline
===================================================================
Self-contained: reads everything from config.yml, no dependency on inference.py.
"""

import os, re, sys, glob, yaml, shutil, argparse, tempfile
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ConcatItemsd, ToTensord, DeleteItemsd,
    MapTransform, Invertd,
)
from monai.data import Dataset, DataLoader

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from utils.nifti_to_rtdose import nifti_to_rtdose_dicom
except ImportError:
    nifti_to_rtdose_dicom = None


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _derive_constants(config):
    RX = float(config["clinical_targets"]["prescription_dose_gy"])
    SPACING = tuple(config["dataset"]["target_spacing"])
    PATCH = tuple(config["dataset"]["patch_size"])
    CHANNELS = [f"{ch["index"]:04d}" for ch in sorted(config["channels"], key=lambda c: c["index"])]
    SIB_KEYS = [lvl["name"] for lvl in config["clinical_targets"]["targets"]]
    SIB_ORDER = sorted(
        [(lvl["name"], lvl["rx_gy"]) for lvl in config["clinical_targets"]["targets"]],
        key=lambda x: x[1],
    )
    BODY_CH_IDX = next(ch["index"] for ch in config["channels"] if ch.get("role") == "Body_Mask")
    M = config.get("model", {})
    return {
        "rx": RX, "spacing": SPACING, "patch": PATCH, "channels": CHANNELS,
        "sib_keys": SIB_KEYS, "sib_order": SIB_ORDER, "body_ch_idx": BODY_CH_IDX,
        "model_channels": tuple(M.get("channels", [16,32,64,128])),
        "model_strides": tuple(M.get("strides", [2,2,2])),
        "model_res_units": int(M.get("num_res_units", 2)),
        "sw_overlap": float(config.get("training",{}).get("val_sw_overlap", 0.25)),
        "sw_batch_size": int(config.get("training",{}).get("val_sw_batch_size", 1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_one(roi_names, patterns):
    for pattern in patterns:
        for name in roi_names:
            if re.match(pattern, name, re.IGNORECASE):
                return name
    return None

def _match_all(roi_names, patterns):
    matched = []
    for name in roi_names:
        for pattern in patterns:
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

def _parse_gantry_angles_inf(plan_files, default_angles):
    if not plan_files:
        print(f"  [pipeline] No RTPlan — defaulting to {len(default_angles)} equispaced gantry angles.")
        return default_angles
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
            print("  [pipeline] No gantry angles — defaulting to fallback angles.")
            return default_angles
        unique_angles = sorted(set(angles))
        print(f"  [pipeline] Gantry Angles: {unique_angles}")
        return unique_angles
    except Exception as e:
        print(f"  [pipeline] RTPlan read error ({e}) — using fallback angles.")
        return default_angles

def _generate_bev_beam_mask(plan_files, ct_image, ptv_mask_array, config):
    _bev = config.get("preprocessing", {}).get("bev", {})
    sad_mm = float(_bev.get("sad_mm", 1000.0))
    penumbra_mm = float(_bev.get("penumbra_mm", 7.0))
    default_angles = list(_bev.get("default_gantry_angles", [0, 51, 102, 154, 205, 257, 308]))
    
    gantry_angles = _parse_gantry_angles_inf(plan_files, default_angles)
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
        source   = iso_phys - beam_dir * sad_mm
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
        radius = np.abs(depth) * np.tan(half_angle) + penumbra_mm
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
# Step 1: Preprocess DICOM → Config-Driven NIfTI
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dicom(dicom_dir: str, images_dir: str, config: dict, case_name: str = "patient_001") -> None:
    os.makedirs(images_dir, exist_ok=True)
    ct_image, rs_ds, plan_files = _scan_dicom_dir(dicom_dir)
    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing  = ct_image.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]
    matched = {}
    masks = {}

    ptv_patterns = config["clinical_targets"].get("ptv_patterns", [])
    if not ptv_patterns:
        for level in config["clinical_targets"].get("targets", []):
            ptv_patterns.extend(level.get("patterns", []))
            
    found_ptvs = _match_all(roi_names, ptv_patterns)
    matched["PTV"] = found_ptvs
    if found_ptvs:
        masks["PTV"] = _union_contours_to_mask(rs_ds, found_ptvs, ct_image)
        print(f"  [preprocess] PTV ({len(found_ptvs)} structures): {found_ptvs}")
    else:
        masks["PTV"] = np.zeros_like(ct_array, dtype=np.uint8)
        print(f"  [preprocess] PTV: NOT FOUND")

    assert masks["PTV"].sum() > 0, "PTV mask is empty — check contour names."

    for oar in config.get("organs_at_risk", []):
        canonical = oar["canonical"]

        if oar.get("split_laterality"):
            l_name = _match_one(roi_names, oar.get("patterns_left", []))
            r_name = _match_one(roi_names, oar.get("patterns_right", []))
            found_names = [n for n in [l_name, r_name] if n]
            matched[canonical] = found_names
            if found_names:
                masks[canonical] = _union_contours_to_mask(rs_ds, found_names, ct_image)
                print(f"  [preprocess] {canonical}: L={l_name or 'NOT FOUND'} R={r_name or 'NOT FOUND'}")
            else:
                masks[canonical] = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"  [preprocess] {canonical}: NOT FOUND — using empty mask")
        else:
            found_name = _match_one(roi_names, oar.get("aliases", []))
            matched[canonical] = found_name
            if found_name:
                masks[canonical] = _contour_to_mask(rs_ds, found_name, ct_image)
                print(f"  [preprocess] {canonical}: {found_name}")
            else:
                masks[canonical] = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"  [preprocess] {canonical}: NOT FOUND — using empty mask")

    sdms = {}
    for oar in config.get("organs_at_risk", []):
        if oar.get("requires_sdm") and oar["canonical"] in masks:
            canonical = oar["canonical"]
            sdms[canonical] = _signed_distance_map(masks[canonical], spacing_zyx)

    body_hu_threshold = float(config.get("body_hu_threshold", -300.0))
    if "Body" in masks and masks["Body"].sum() == 0:
        print(f"  [preprocess] No Body contour — falling back to HU threshold ({body_hu_threshold} HU)")
        masks["Body"] = (ct_array > body_hu_threshold).astype(np.float32)

    beam_mask = None
    has_bev = any(ch.get("role") == "BEV_Beam" for ch in config.get("channels", []))
    if has_bev:
        beam_mask = _generate_bev_beam_mask(plan_files, ct_image, masks["PTV"], config)

    def _save(arr, suffix):
        path = os.path.join(images_dir, f"{case_name}_{suffix}.nii.gz")
        sitk.WriteImage(_numpy_to_sitk(arr.astype(np.float32), ct_image), path)
        return path

    for ch in config.get("channels", []):
        idx = f"{ch['index']:04d}"
        role = ch.get("role")
        organ = ch.get("organ")

        if role == "CT":
            _save(ct_array, idx)
        elif role == "PTV_binary":
            _save(masks["PTV"], idx)
        elif role == "BEV_Beam":
            _save(beam_mask, idx)
        elif role == "Body_Mask":
            _save(masks.get("Body", (ct_array > body_hu_threshold).astype(np.float32)), idx)
        elif role == "SDM" and organ in sdms:
            _save(sdms[organ], idx)
        elif role == "Binary_Mask" and organ in masks:
            _save(masks[organ], idx)

    targets = config["clinical_targets"].get("targets", [])
    if targets:
        individual_masks = {}
        for p_name in matched.get("PTV", []):
            individual_masks[p_name] = _contour_to_mask(rs_ds, p_name, ct_image)
            
        accumulated = {}
        for p_name, p_mask in individual_masks.items():
            canonical_target = None
            for level in targets:
                patterns = level.get("patterns", [])
                if any(re.match(pat, p_name, re.IGNORECASE) for pat in patterns):
                    canonical_target = level["name"]
                    break
            
            if canonical_target is None:
                canonical_target = p_name.replace(" ", "_").replace("/", "_")
            
            if canonical_target in accumulated:
                accumulated[canonical_target] = np.maximum(accumulated[canonical_target], p_mask)
            else:
                accumulated[canonical_target] = p_mask

        for canon_name, p_mask in accumulated.items():
            _save(p_mask.astype(np.float32), canon_name)

    print(f"  [preprocess] Channels saved → {images_dir}")


# --- Config-driven Painter's Algorithm (matches training.py exactly) ---
class CreateDiscretePTVMapd(MapTransform):
    def __init__(self, keys, sib_order, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.processing_order = sib_order  # sorted lowest-to-highest rx_gy

    def __call__(self, data):
        d = dict(data)
        discrete_ptv = torch.zeros_like(d["ch_0"])
        for p_key, dose_val in self.processing_order:
            if p_key in d:
                mask = d[p_key] >= 0.5
                discrete_ptv = torch.where(
                    mask,
                    torch.tensor(dose_val, dtype=discrete_ptv.dtype, device=discrete_ptv.device),
                    discrete_ptv,
                )
                del d[p_key]
        d["discrete_ptv"] = discrete_ptv
        return d


def _build_inference_transforms(C, config):
    """
    Build the deterministic inference transform chain that exactly mirrors
    val_transforms from training.py.

    Channel ordering in the concatenated 'image' tensor:
      0 = CT (normalized HU)           — ch_0  bilinear → NormalizeIntensityd
      1 = discrete PTV map (SIB Rx Gy) — discrete_ptv  ← CreateDiscretePTVMapd
      2 = Bladder SDM                  — ch_2  bilinear
      3 = Anorectum SDM                — ch_3  bilinear
      4 = Body Mask                    — ch_4  nearest
      5 = Penile Bulb binary           — ch_5  nearest
      6 = BEV Beam Frustum             — ch_6  nearest
    """
    sib_keys = C["sib_keys"]

    # ── Channel keys that are loaded from disk ────────────────────────────────
    # ch_0 … ch_N from config, plus individual PTV structure files for SIB map
    ch_keys = [f"ch_{ch['index']}" for ch in sorted(config["channels"], key=lambda c: c["index"])]
    # Add SIB sub-volume keys (e.g. PTV_62_20, PTV_44_20)
    all_keys = ch_keys + sib_keys

    # ── Spacingd interpolation modes — derived from config (never hardcoded) ──
    # Config channels sorted by index → same order as ch_keys
    ch_modes = tuple(
        ch["interpolation"]
        for ch in sorted(config["channels"], key=lambda c: c["index"])
    ) + ("nearest",) * len(sib_keys)   # SIB masks always nearest

    # ── Keys that exist after CreateDiscretePTVMapd (SIB keys are deleted) ───
    # The individual PTV files (PTV_62_20, PTV_44_20 …) are consumed and
    # removed by CreateDiscretePTVMapd; do NOT list them in DeleteItemsd.
    # ch_1 (PTV_binary) is intentionally kept in the dict so that it is
    # available to the body-mask and ring-mask logic in post-processing;
    # it is NOT passed to the model (discrete_ptv replaces it in ConcatItemsd).
    model_concat_keys = [
        f"ch_{ch['index']}" for ch in sorted(config["channels"], key=lambda c: c["index"])
        if ch["key"] != "ch_1"        # skip raw PTV_binary — replaced by discrete_ptv
    ]
    # Build final concat list: [ch_0, discrete_ptv, ch_2, ch_3, ch_4, ch_5, ch_6]
    concat_keys = [ch_keys[0], "discrete_ptv"] + [
        k for k in ch_keys[1:] if k != "ch_1"
    ]

    # Keys to clean up after concat.
    # IMPORTANT: ch_0 is intentionally KEPT in the dict so that Invertd
    # can read its Spacingd applied_operations trace in post-processing.
    # ch_1 is already gone (consumed by CreateDiscretePTVMapd).
    delete_keys = [k for k in ch_keys if k not in ("ch_0", "ch_1")] + ["discrete_ptv"]

    return Compose([
        # 1. Load every NIfTI from disk
        LoadImaged(keys=all_keys, allow_missing_keys=True),
        # 2. Ensure (C, D, H, W) layout
        EnsureChannelFirstd(keys=all_keys, allow_missing_keys=True),
        # 3. Resample everything to target_spacing — this is the op we must
        #    invert later to get back to native CT geometry
        Spacingd(keys=all_keys, pixdim=C["spacing"], mode=ch_modes,
                 allow_missing_keys=True),
        # 4. Paint SIB dose levels onto a single discrete map (Painter's Algo)
        #    This step DELETES the individual PTV_62_20, PTV_44_20 … keys.
        CreateDiscretePTVMapd(keys=["ch_0"], sib_order=C["sib_order"]),
        # 5. Z-score normalise CT HU values — MUST happen after Spacingd,
        #    before ConcatItemsd, exactly as in val_transforms.
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        # 6. Stack into a single 7-channel tensor
        #    Channel order: [CT_norm, discrete_ptv, bladder_sdm, ano_sdm,
        #                    body_mask, penile_bulb, bev_beam]
        ConcatItemsd(keys=concat_keys, name="image"),
        # 7. Clean up individual channel keys (ch_1 was already removed above)
        DeleteItemsd(keys=delete_keys),
        # 8. Ensure the output is a torch.Tensor
        ToTensord(keys=["image"]),
    ])


def run_inference(patient_id, images_dir, config, C, output_dir=".",
                  model_path=None, save_nifti=True):
    """
    Run sliding-window inference on one patient.

    Spatial flow
    ============
    Native CT  →  [Spacingd @ target_spacing]  →  Model  →  [body-mask]  →
    NIfTI saved at target_spacing  →  [nifti_to_rtdose resamples back to
    native CT grid]  →  RTDOSE DICOM

    The NIfTI is saved with the exact origin / direction / spacing of the
    resampled grid so that SimpleITK can later resample it back to the
    native CT geometry without any affine mismatch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n[inference] Loading model on {device} ...")
    model = UNet(
        spatial_dims=3, in_channels=7, out_channels=1,
        channels=C["model_channels"], strides=C["model_strides"],
        num_res_units=C["model_res_units"],
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    ckpt = torch.load(model_path, map_location=device)
    # Support both raw state-dict and wrapped checkpoint dicts
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[inference] Weights loaded from {model_path}")

    # ── 2. Build per-patient input dict ──────────────────────────────────────
    pt_dict = {}
    # Channel NIfTIs (ch_0 … ch_N)
    for ch_cfg in sorted(config["channels"], key=lambda c: c["index"]):
        i = ch_cfg["index"]
        ch_path = os.path.join(images_dir, f"{patient_id}_{i:04d}.nii.gz")
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Input channel not found: {ch_path}")
        pt_dict[f"ch_{i}"] = ch_path

    # Individual SIB PTV structure NIfTIs (e.g. PTV_62_20, PTV_44_20)
    for ptv_key in C["sib_keys"]:
        p = os.path.join(images_dir, f"{patient_id}_{ptv_key}.nii.gz")
        if os.path.exists(p):
            pt_dict[ptv_key] = p
        else:
            print(f"[inference] WARNING: SIB key '{ptv_key}' not found — "
                  f"discrete_ptv will not include it.")

    # ── 3. Cache the native CT sitk image for inverse-transform metadata ──────
    # We record origin/direction/spacing of the target-spacing grid so that
    # the saved NIfTI carries the correct spatial metadata for nifti_to_rtdose.
    native_ct_sitk = sitk.ReadImage(pt_dict["ch_0"])
    native_spacing  = native_ct_sitk.GetSpacing()   # XYZ
    native_size     = native_ct_sitk.GetSize()       # XYZ
    native_origin   = native_ct_sitk.GetOrigin()
    native_direction = native_ct_sitk.GetDirection()

    # Compute the resampled grid size (same formula used by Spacingd internally)
    resampled_size = [
        int(round(native_size[i] * native_spacing[i] / C["spacing"][i]))
        for i in range(3)
    ]
    resampled_spacing = tuple(C["spacing"])   # (x, y, z) in mm

    print(f"[inference] Native CT  : size={native_size}  spacing={native_spacing}")
    print(f"[inference] Resampled  : size={resampled_size}  spacing={resampled_spacing}")

    # ── 4. Apply inference transforms (mirrors val_transforms exactly) ────────
    transforms = _build_inference_transforms(C, config)
    ds     = Dataset(data=[pt_dict], transform=transforms)
    loader = DataLoader(ds, batch_size=1)
    batch  = next(iter(loader))
    inputs = batch["image"].to(device)

    print(f"[inference] Input tensor shape : {tuple(inputs.shape)}  "
          f"(B, 7, D, H, W) — at target spacing")

    # ── 5. Sliding-window inference ───────────────────────────────────────────
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=C["patch"],
                sw_batch_size=C["sw_batch_size"],
                predictor=model,
                overlap=C["sw_overlap"],
                mode="gaussian",
            )

    # ── 6. Post-inference: softplus + body mask + Gy scaling ─────────────────
    # All done together in target-spacing space before Invertd.
    outputs      = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
    outputs_gy   = F.softplus(outputs.float()) * C["rx"]

    _BODY_CH_IDX = C["body_ch_idx"]
    body_mask    = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX + 1, ...] > 0.5).float()
    masked_dose  = torch.clamp(outputs_gy * body_mask, min=0.0)  # (1, 1, D, H, W)

    print(f"[inference] Body-masked dose (target spacing): shape={tuple(masked_dose.shape)}"
          f"  range=[{masked_dose.min():.2f}, {masked_dose.max():.2f}] Gy")

    # ── 9. Inverse spatial transform via MONAI Invertd ───────────────────────────
    # ch_0 was deliberately kept in the batch (not deleted by DeleteItemsd).
    # It is a MetaTensor whose applied_operations list records the exact Spacingd
    # transform that was applied during preprocessing.  Invertd reads those
    # breadcrumbs and reverses Spacingd to restore the native CT grid geometry.
    try:
        from monai.data import MetaTensor as _MetaTensor

        # Work with a single (un-batched) item for Invertd
        ch0_batched = batch["ch_0"]           # (1, 1, D, H, W) MetaTensor
        ch0_single  = ch0_batched[0]          # (1, D, H, W)  — one sample

        # Build pred_dose MetaTensor: copy ch_0's spatial metadata so that
        # Invertd has the Spacingd trace attached to it.
        pred_single = masked_dose[0].cpu()    # (1, D, H, W)
        if isinstance(ch0_single, _MetaTensor):
            pred_meta = _MetaTensor(pred_single, meta=ch0_single.meta.copy())
        else:
            # Older MONAI without MetaTensor — fall through to sitk fallback
            raise RuntimeError("batch['ch_0'] is not a MetaTensor")

        invert_data = {
            "ch_0"     : ch0_single,   # provides the Spacingd trace
            "pred_dose": pred_meta,    # carries the same trace → Invertd reverses it
        }

        inverter = Invertd(
            keys         = "pred_dose",
            transform    = transforms,   # the full preprocessing Compose chain
            orig_keys    = "ch_0",       # CRITICAL: read Spacingd trace from ch_0
            nearest_interp = False,      # linear interpolation for dose
            to_tensor    = True,
        )

        inverted        = inverter(invert_data)
        final_dose_mt   = inverted["pred_dose"]  # MetaTensor at native CT geometry

        # Extract numpy array — channel-first after inversion, so squeeze channel 0
        pred_arr = final_dose_mt.numpy() if hasattr(final_dose_mt, "numpy") \
                   else final_dose_mt.cpu().numpy()
        if pred_arr.ndim == 4:   # (1, D, H, W) → (D, H, W)
            pred_arr = pred_arr[0]
        pred_dose_native_np = np.clip(pred_arr.astype(np.float32), 0.0, None)

        print(f"[inference] Invertd succeeded.")
        print(f"[inference]   Shape at native grid : {pred_dose_native_np.shape}")
        _used_invertd = True

    except Exception as _exc:
        # ── Fallback: sitk ResampleImageFilter ────────────────────────────────
        # If Invertd fails (old MONAI / missing MetaTensor), resample the
        # target-spacing prediction back to the native CT grid via sitk.
        print(f"[inference] WARNING: Invertd failed ({_exc}). "
              f"Falling back to sitk ResampleImageFilter.")
        _used_invertd = False

        pred_np_tgt = masked_dose[0, 0].cpu().numpy().astype(np.float32)  # (D,H,W)
        pred_sitk_tgt = sitk.GetImageFromArray(pred_np_tgt)
        pred_sitk_tgt.SetSpacing(resampled_spacing)
        pred_sitk_tgt.SetOrigin(native_origin)
        pred_sitk_tgt.SetDirection(native_direction)

        fb = sitk.ResampleImageFilter()
        fb.SetReferenceImage(native_ct_sitk)
        fb.SetInterpolator(sitk.sitkLinear)
        fb.SetDefaultPixelValue(0.0)
        pred_dose_native_np = np.clip(
            sitk.GetArrayFromImage(fb.Execute(pred_sitk_tgt)).astype(np.float32),
            0.0, None,
        )

    print(f"[inference]   Native grid shape   : {pred_dose_native_np.shape}")
    print(f"[inference]   Dose range (native) : "
          f"[{pred_dose_native_np.min():.2f}, {pred_dose_native_np.max():.2f}] Gy")

    # ── 10. Save NIfTI anchored to native CT geometry ────────────────────────
    # Use native_ct_sitk.CopyInformation() as a hard anchor: this copies the
    # origin, direction, and spacing from the original CT DICOM — so even if
    # Invertd's affine has a tiny floating-point residual, the saved NIfTI is
    # guaranteed to register perfectly with the CT.
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{patient_id}_predicted_dose.nii.gz")
        dose_sitk = sitk.GetImageFromArray(pred_dose_native_np)
        dose_sitk.CopyInformation(native_ct_sitk)  # hard-anchor: origin/dir/spacing
        sitk.WriteImage(dose_sitk, out_file)
        method = "Invertd" if _used_invertd else "sitk-fallback"
        print(f"[inference] Saved NIfTI (native grid, {method}) → {out_file}")

    metadata = {
        "patient_id"     : patient_id,
        "spacing"        : native_ct_sitk.GetSpacing(),
        "origin"         : native_ct_sitk.GetOrigin(),
        "direction"      : native_ct_sitk.GetDirection(),
        "shape"          : pred_dose_native_np.shape,
        "dose_min"       : float(pred_dose_native_np.min()),
        "dose_max"       : float(pred_dose_native_np.max()),
        "target_spacing" : resampled_spacing,
        "used_invertd"   : _used_invertd if "_used_invertd" in dir() else False,
    }
    return pred_dose_native_np, metadata


def run_pipeline(dicom_dir, config_path="config.yml", model_path=None,
                 dose_spacing_mm=None, keep_temp=False):
    config = load_config(config_path)
    C = _derive_constants(config)

    print("\n" + "="*50)
    print("  RADIOTHERAPY DOSE PREDICTION PIPELINE")
    print("="*50)

    tmp_dir = tempfile.mkdtemp(prefix="dose_infer_")
    images_dir = os.path.join(tmp_dir, "imagesTr")
    os.makedirs(images_dir, exist_ok=True)
    print(f"\n[1] Workspace created: {tmp_dir}")

    print(f"\n[2] Preprocessing DICOM...")
    preprocess_dicom(dicom_dir, images_dir, config, case_name="case_infer")

    print(f"\n[3] Running Neural Network Inference...")
    pred_dose, metadata = run_inference(
        patient_id="case_infer", images_dir=images_dir,
        config=config, C=C, output_dir=tmp_dir,
        model_path=model_path, save_nifti=True,
    )

    nifti_dose_file = os.path.join(tmp_dir, "case_infer_predicted_dose.nii.gz")

    print(f"\n[4] Building RTDOSE DICOM...")
    if dose_spacing_mm is None:
        dose_spacing_mm = C["spacing"][2]

    rtdose_path = None
    if nifti_to_rtdose_dicom is not None:
        try:
            rtdose_path = nifti_to_rtdose_dicom(
                ct_rs_dir=dicom_dir, nifti_path=nifti_dose_file,
                dose_spacing_mm=dose_spacing_mm,
            )
        except Exception as e:
            print(f"  RTDOSE generation failed: {e}")
    else:
        print("  nifti_to_rtdose_dicom not available. Skipped.")

    if keep_temp:
        print(f"\n[5] Retaining workspace: {tmp_dir}")
    else:
        print("\n[5] Cleaning up workspace...")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "="*50)
    print("  PIPELINE COMPLETE")
    print("="*50)
    if rtdose_path:
        print(f"  RTDOSE saved to: {rtdose_path}")
    return rtdose_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end dose prediction pipeline")
    parser.add_argument("--dicom-dir", required=True, type=str)
    parser.add_argument("--config", default="config.yml", type=str)
    parser.add_argument("--model", default="best_dose_model_clinical_jun3.pth", type=str)
    parser.add_argument("--dose-spacing", default=None, type=float)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        dicom_dir=args.dicom_dir, config_path=args.config,
        model_path=args.model, dose_spacing_mm=args.dose_spacing,
        keep_temp=args.keep_temp,
    )
