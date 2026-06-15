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
    NormalizeIntensityd, ConcatItemsd, ToTensord, DeleteItemsd, MapTransform,
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
    sib_keys = C["sib_keys"]
    ch_keys = ["ch_0","ch_1","ch_2","ch_3","ch_4","ch_5","ch_6"] + sib_keys
    ch_modes = ("bilinear","nearest","bilinear","bilinear","nearest","nearest","nearest") + ("nearest",)*len(sib_keys)
    return Compose([
        LoadImaged(keys=ch_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=ch_keys, allow_missing_keys=True),
        Spacingd(keys=ch_keys, pixdim=C["spacing"], mode=ch_modes, allow_missing_keys=True),
        CreateDiscretePTVMapd(keys=["ch_0"], sib_order=C["sib_order"]),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0","discrete_ptv","ch_2","ch_3","ch_4","ch_5","ch_6"], name="image"),
        DeleteItemsd(keys=["ch_0","ch_1","ch_2","ch_3","ch_4","ch_5","ch_6"]),
        ToTensord(keys=["image"]),
    ])


def run_inference(patient_id, images_dir, config, C, output_dir=".",
                  model_path=None, save_nifti=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    model = UNet(
        spatial_dims=3, in_channels=7, out_channels=1,
        channels=C["model_channels"], strides=C["model_strides"],
        num_res_units=C["model_res_units"],
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model weights loaded from {model_path}")

    # Build input dict
    pt_dict = {}
    for i, ch in enumerate(C["channels"]):
        ch_path = os.path.join(images_dir, f"{patient_id}_{ch}.nii.gz")
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Input file not found: {ch_path}")
        pt_dict[f"ch_{i}"] = ch_path

    # Individual PTV files
    for ptv_key in C["sib_keys"]:
        p = os.path.join(images_dir, f"{patient_id}_{ptv_key}.nii.gz")
        if os.path.exists(p):
            pt_dict[ptv_key] = p

    transforms = _build_inference_transforms(C, config)
    ds = Dataset(data=[pt_dict], transform=transforms)
    loader = DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    inputs = batch["image"].to(device)

    # Spatial metadata from resampled CT
    ct_sitk = sitk.ReadImage(pt_dict["ch_0"])
    ct_resampled = sitk.Resample(
        ct_sitk,
        [int(round(ct_sitk.GetSize()[i] * ct_sitk.GetSpacing()[i] / C["spacing"][i])) for i in range(3)],
        sitk.Transform(), sitk.sitkLinear, ct_sitk.GetOrigin(),
        C["spacing"], ct_sitk.GetDirection(), 0.0, ct_sitk.GetPixelID(),
    )
    grid_origin = ct_resampled.GetOrigin()
    grid_direction = ct_resampled.GetDirection()

    print(f"Input tensor shape: {inputs.shape}")

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = sliding_window_inference(
                inputs=inputs, roi_size=C["patch"],
                sw_batch_size=C["sw_batch_size"], predictor=model,
                overlap=C["sw_overlap"], mode="gaussian",
            )

    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
    outputs_activated = F.softplus(outputs.float())

    body_idx = C["body_ch_idx"]
    body_mask_hard = (inputs[:, body_idx:body_idx+1, ...] > 0.5).float()
    outputs_activated = outputs_activated * body_mask_hard

    pred_dose = outputs_activated[0, 0].cpu().numpy()
    pred_dose = pred_dose * C["rx"]
    pred_dose = np.clip(pred_dose, 0.0, None)

    print(f"Prediction complete. Shape: {pred_dose.shape}")
    print(f"Dose range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")

    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{patient_id}_predicted_dose.nii.gz")
        sitk_img = sitk.GetImageFromArray(pred_dose.astype(np.float32))
        sitk_img.SetSpacing(C["spacing"])
        sitk_img.SetOrigin(grid_origin)
        sitk_img.SetDirection(grid_direction)
        sitk.WriteImage(sitk_img, out_file)
        print(f"Saved predicted dose to: {out_file}")

    metadata = {
        "patient_id": patient_id, "spacing": C["spacing"],
        "origin": grid_origin, "direction": grid_direction,
        "shape": pred_dose.shape,
        "dose_min": float(pred_dose.min()), "dose_max": float(pred_dose.max()),
    }
    return pred_dose, metadata


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
