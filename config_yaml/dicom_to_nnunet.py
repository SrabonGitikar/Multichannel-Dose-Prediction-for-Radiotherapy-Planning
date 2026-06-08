"""
DICOM → nnU-Net v2 Converter
==============================
Converts raw DICOM data (CT + RTStruct + RTPlan + RTDose) into the
nnU-Net v2 regression format.

All site-specific knowledge (structure names, channel layout, paths,
BEV parameters, SIB canonical mapping) is read from config.yml.
This script contains only DICOM I/O and preprocessing logic.
"""

import os
import re
import json
import yaml
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pydicom
# pyrefly: ignore [missing-import]
import SimpleITK as sitk
# pyrefly: ignore [missing-import]
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon

# ===========================================================================
# CONFIGURATION
# ===========================================================================

with open("config.yml", "r") as _f:
    config = yaml.safe_load(_f)

_pre  = config["preprocessing"]
_bev  = _pre["bev"]

DATA_DIR      = _pre["dicom_dir"]
OUTPUT_DIR    = _pre["output_dir"]
CASE_PREFIX   = _pre["case_prefix"]
DATASET_NAME  = _pre["dataset_name"]

SAD_MM              = float(_bev["sad_mm"])
PENUMBRA_MM         = float(_bev["penumbra_mm"])
DEFAULT_GANTRY_ANGLES = list(_bev["default_gantry_angles"])

BODY_HU_THRESHOLD   = float(config["body_hu_threshold"])

# Build channel-name map for dataset.json from config
_CHANNEL_NAMES = {
    str(ch["index"]): ch.get("organ", ch["role"])
    for ch in sorted(config["channels"], key=lambda c: c["index"])
}

# Structure-matching rules indexed by canonical name for fast access
_STRUCT_RULES = {s["canonical"]: s for s in _pre["structure_matching"]}


# ===========================================================================
# STRUCTURE MATCHING — driven entirely by config
# ===========================================================================

def _match_one(roi_names, patterns):
    """Return the first roi_name matching any pattern (case-insensitive)."""
    for pattern in patterns:
        for name in roi_names:
            if re.match(pattern, name, re.IGNORECASE):
                return name
    return None

def _match_all(roi_names, patterns):
    """Return ALL roi_names matching any pattern (case-insensitive)."""
    matched = []
    for name in roi_names:
        for pattern in patterns:
            if re.match(pattern, name, re.IGNORECASE):
                matched.append(name)
                break
    return matched


# ===========================================================================
# DICOM I/O HELPERS
# ===========================================================================

def find_dicom_subdir(patient_dir):
    subdirs = [d for d in os.listdir(patient_dir)
               if os.path.isdir(os.path.join(patient_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(patient_dir, subdirs[0])
    return patient_dir

def sort_dicom_files(dicom_dir):
    """Sort files in a directory by DICOM Modality tag."""
    sorted_files = {"RTSTRUCT": [], "RTPLAN": [], "RTDOSE": []}
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if hasattr(ds, "Modality") and ds.Modality in sorted_files:
                sorted_files[ds.Modality].append(fpath)
        except Exception:
            pass
    return sorted_files

def find_correct_rtstruct(plan_files, struct_files):
    if not struct_files:
        raise FileNotFoundError("No RTStruct files found in directory!")
    if len(struct_files) == 1:
        return struct_files[0]
    if not plan_files:
        print("    WARNING: Multiple RTStructs, no RTPlan found — using first.")
        return struct_files[0]
    plan_ds = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
    ref_uid = None
    if hasattr(plan_ds, "ReferencedStructureSetSequence"):
        ref_uid = plan_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
    for sf in struct_files:
        ds = pydicom.dcmread(sf, stop_before_pixels=True)
        if ds.SOPInstanceUID == ref_uid:
            return sf
    print("    WARNING: No RTStruct matched RTPlan reference — using first.")
    return struct_files[0]


# ===========================================================================
# CONTOUR RASTERISATION & SDM
# ===========================================================================

def rtstruct_contour_to_mask(rs_ds, roi_name, ct_image):
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
    for roi_contour in rs_ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            contour_seq = roi_contour
            break
    if contour_seq is None or not hasattr(contour_seq, "ContourSequence"):
        return mask
    for contour in contour_seq.ContourSequence:
        points = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        pixel_coords = []
        for pt in points:
            idx = ct_image.TransformPhysicalPointToContinuousIndex(
                (float(pt[0]), float(pt[1]), float(pt[2]))
            )
            pixel_coords.append(idx)
        pixel_coords = np.array(pixel_coords)
        slice_idx = int(round(pixel_coords[0, 2]))
        if 0 <= slice_idx < mask.shape[0]:
            rr, cc = polygon(pixel_coords[:, 1], pixel_coords[:, 0],
                             shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1
    return mask

def rtstruct_union_mask(rs_ds, roi_name_list, ct_image):
    """Union-rasterize a list of contour names into one binary mask."""
    ct_array = sitk.GetArrayFromImage(ct_image)
    union = np.zeros(ct_array.shape, dtype=np.uint8)
    for name in roi_name_list:
        single = rtstruct_contour_to_mask(rs_ds, name, ct_image)
        union = np.maximum(union, single)
    return union

def compute_signed_distance_map(binary_mask, spacing_mm):
    if binary_mask.sum() == 0:
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0
    dist_outside = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)
    dist_inside  = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)
    return (dist_outside - dist_inside).astype(np.float32)


# ===========================================================================
# RTDose / utility
# ===========================================================================

def load_rtdose_as_sitk(dose_files, ct_image):
    if not dose_files:
        raise FileNotFoundError("No RTDose file found!")
    ds = pydicom.dcmread(dose_files[0])
    dose_array = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_origin  = [float(x) for x in ds.ImagePositionPatient]
    dose_spacing = [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]),
                    float(ds.GridFrameOffsetVector[1]) - float(ds.GridFrameOffsetVector[0])]
    dose_image.SetOrigin(dose_origin)
    dose_image.SetSpacing(dose_spacing)
    dose_image.SetDirection(ct_image.GetDirection())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(dose_image)

def numpy_to_sitk(array, reference_image):
    img = sitk.GetImageFromArray(array)
    img.SetOrigin(reference_image.GetOrigin())
    img.SetSpacing(reference_image.GetSpacing())
    img.SetDirection(reference_image.GetDirection())
    return img


# ===========================================================================
# BEV FRUSTUM BEAM MASK GENERATION  — parameters from config.preprocessing.bev
# ===========================================================================

def _parse_gantry_angles(plan_files):
    if not plan_files:
        print("         WARNING: No RTPlan found. Using default gantry angles.")
        return DEFAULT_GANTRY_ANGLES
    try:
        plan = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
        angles = []
        if hasattr(plan, "BeamSequence"):
            for beam in plan.BeamSequence:
                d_type = getattr(beam, "TreatmentDeliveryType", "")
                b_type = getattr(beam, "BeamType", "")
                if d_type == "TREATMENT" or b_type == "STATIC":
                    if hasattr(beam, "ControlPointSequence") and beam.ControlPointSequence:
                        cp0 = beam.ControlPointSequence[0]
                        if hasattr(cp0, "GantryAngle"):
                            angles.append(float(cp0.GantryAngle))
        if not angles:
            print("         WARNING: No gantry angles in RTPlan. Using defaults.")
            return DEFAULT_GANTRY_ANGLES
        unique_angles = sorted(set(angles))
        print(f"         Gantry angles from RTPlan: {unique_angles}")
        return unique_angles
    except Exception as e:
        print(f"         WARNING: Could not read RTPlan ({e}). Using defaults.")
        return DEFAULT_GANTRY_ANGLES

def generate_bev_beam_mask(plan_files, ct_image, ptv_mask_array):
    """
    Build a 3-D binary BEV frustum mask using parameters from config.yml.
    SAD and penumbra margin are read from config.preprocessing.bev.
    """
    gantry_angles = _parse_gantry_angles(plan_files)
    shape_zyx = sitk.GetArrayViewFromImage(ct_image).shape
    origin    = np.array(ct_image.GetOrigin())
    spacing   = np.array(ct_image.GetSpacing())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    zi, yi, xi = np.meshgrid(
        np.arange(shape_zyx[0]),
        np.arange(shape_zyx[1]),
        np.arange(shape_zyx[2]),
        indexing="ij",
    )
    voxel_indices = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)
    phys_pts = origin + (voxel_indices * spacing) @ direction.T

    ptv_z, ptv_y, ptv_x = np.where(ptv_mask_array > 0.5)
    if len(ptv_z) == 0:
        print("         WARNING: PTV mask empty — returning blank beam mask.")
        return np.zeros(shape_zyx, dtype=np.float32)

    iso_idx  = np.array([ptv_x.mean(), ptv_y.mean(), ptv_z.mean()])
    iso_phys = origin + (iso_idx * spacing) @ direction.T
    print(f"         PTV isocenter (mm): {np.round(iso_phys, 1)}")

    beam_flat = np.zeros(len(phys_pts), dtype=np.float32)

    for angle_deg in gantry_angles:
        theta    = np.deg2rad(angle_deg)
        beam_dir = np.array([np.sin(theta), -np.cos(theta), 0.0])
        source_pos = iso_phys - beam_dir * SAD_MM

        vec   = phys_pts - source_pos
        depth = vec @ beam_dir
        proj_along = depth[:, np.newaxis] * beam_dir
        perp_dist  = np.linalg.norm(vec - proj_along, axis=1)

        ptv_indices = np.stack([ptv_x, ptv_y, ptv_z], axis=1)
        ptv_phys    = origin + (ptv_indices * spacing) @ direction.T
        vec_ptv     = ptv_phys - source_pos
        depth_ptv   = vec_ptv @ beam_dir
        perp_ptv    = np.linalg.norm(
            vec_ptv - depth_ptv[:, np.newaxis] * beam_dir, axis=1
        )

        if depth_ptv.max() > 0:
            half_angle = np.arctan2(perp_ptv.max(), depth_ptv[depth_ptv > 0].min())
        else:
            half_angle = np.deg2rad(5.0)

        radius_at_voxel = np.abs(depth) * np.tan(half_angle) + PENUMBRA_MM
        inside = (depth > 0) & (perp_dist <= radius_at_voxel)
        beam_flat[inside] = 1.0

    beam_zyx = beam_flat.reshape(shape_zyx)
    print(f"         BEV Beam Mask Voxels ({len(gantry_angles)} beams): {int(beam_zyx.sum()):,}")
    return beam_zyx


# ===========================================================================
# MAIN CONVERTER — config-driven structure matching and channel writing
# ===========================================================================

def convert_patient(patient_dir, case_id, images_dir, labels_dir):
    pid = os.path.basename(patient_dir)[:12]
    case_name = f"{CASE_PREFIX}_{case_id:03d}"
    print(f"\n{'='*60}")
    print(f"  Converting: {pid}  (case: {case_name})")
    print(f"{'='*60}")

    # ---- Load CT -------------------------------------------------------
    dicom_dir   = find_dicom_subdir(patient_dir)
    dicom_files = sort_dicom_files(dicom_dir)
    plan_files  = dicom_files.get("RTPLAN", [])
    struct_files = dicom_files.get("RTSTRUCT", [])
    dose_files  = dicom_files.get("RTDOSE", [])

    print("  [1/8] Loading CT volume...")
    reader     = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print("  *** SKIPPING: No DICOM series found ***")
        return False
    best_series, best_count = None, 0
    for sid in series_ids:
        fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(fnames) > best_count:
            best_count, best_series = len(fnames), fnames
    reader.SetFileNames(best_series)
    ct_image  = reader.Execute()
    ct_array  = sitk.GetArrayFromImage(ct_image)
    spacing   = ct_image.GetSpacing()
    print(f"         Shape: {ct_array.shape}, Spacing: {spacing}")
    if ct_image.GetDimension() != 3:
        print(f"  *** SKIPPING: CT is {ct_image.GetDimension()}D ***")
        return False

    # ---- Parse RTStruct ------------------------------------------------
    print("  [2/8] Parsing RTStruct contours...")
    rs_file = find_correct_rtstruct(plan_files, struct_files)
    rs_ds   = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]

    # ---- Match all structures from config ------------------------------
    print("  [3/8] Matching structures from config...")
    matched   = {}   # canonical → matched roi_name(s)
    masks     = {}   # canonical → numpy uint8 array
    skip      = False

    for rule in _pre["structure_matching"]:
        canonical = rule["canonical"]
        if rule.get("generated"):
            continue   # BEV handled separately

        if rule.get("split_laterality"):
            l_name = _match_one(roi_names, rule["patterns_left"])
            r_name = _match_one(roi_names, rule["patterns_right"])
            found_names = [n for n in [l_name, r_name] if n]
            matched[canonical] = found_names
            if found_names:
                masks[canonical] = rtstruct_union_mask(rs_ds, found_names, ct_image)
                print(f"         {canonical}: L={l_name or 'NOT FOUND'}  "
                      f"R={r_name or 'NOT FOUND'}  "
                      f"({masks[canonical].sum():,} voxels)")
            else:
                masks[canonical] = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"         {canonical}: NOT FOUND — using empty mask")

        elif rule.get("merge"):
            found_names = _match_all(roi_names, rule["patterns"])
            matched[canonical] = found_names
            if found_names:
                masks[canonical] = rtstruct_union_mask(rs_ds, found_names, ct_image)
                print(f"         {canonical} ({len(found_names)} structures): {found_names}")
            else:
                masks[canonical] = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"         {canonical}: NOT FOUND")
        else:
            found_name = _match_one(roi_names, rule["patterns"])
            matched[canonical] = found_name
            if found_name:
                masks[canonical] = rtstruct_contour_to_mask(rs_ds, found_name, ct_image)
                print(f"         {canonical}: {found_name}  ({masks[canonical].sum():,} voxels)")
            else:
                masks[canonical] = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"         {canonical}: NOT FOUND — using empty mask")

        # Enforce required structures
        if rule.get("required") and masks[canonical].sum() == 0:
            print(f"  *** SKIPPING: required structure '{canonical}' is empty or absent ***")
            skip = True

    if skip:
        return False

    ptv_mask = masks["PTV"]
    if ptv_mask.sum() == 0:
        print("  *** SKIPPING: PTV mask is empty ***")
        return False

    # ---- SDMs for channels that need them ------------------------------
    print("  [4/8] Computing signed distance maps...")
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    sdms = {}
    for rule in _pre["structure_matching"]:
        if rule.get("compute_sdm") and rule["canonical"] in masks:
            canonical = rule["canonical"]
            sdms[canonical] = compute_signed_distance_map(masks[canonical], spacing_zyx)
            print(f"         SDM computed for {canonical}")

    # ---- Body mask (with HU-threshold fallback) ------------------------
    print("  [5/8] Computing Body Mask...")
    body_mask = masks.get("Body", np.zeros_like(ct_array, dtype=np.uint8))
    if body_mask.sum() == 0:
        print(f"         No Body contour — falling back to HU threshold ({BODY_HU_THRESHOLD} HU)")
        body_mask = (ct_array > BODY_HU_THRESHOLD).astype(np.float32)
    else:
        body_mask = body_mask.astype(np.float32)
    print(f"         Body Mask Voxels: {int(body_mask.sum()):,}")

    # ---- Load RTDose ---------------------------------------------------
    print("  [6/8] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dose_files, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)

    # ---- BEV Beam Frustum Mask -----------------------------------------
    print("  [7/8] Generating BEV Frustum Beam Mask...")
    beam_mask = generate_bev_beam_mask(plan_files, ct_image, ptv_mask)

    # ---- Write NIfTI files from channel layout in config ---------------
    print("  [8/8] Saving NIfTI files...")

    def _write(arr, suffix):
        path = os.path.join(images_dir, f"{case_name}_{suffix}.nii.gz")
        sitk.WriteImage(numpy_to_sitk(arr.astype(np.float32), ct_image), path)

    for rule in _pre["structure_matching"]:
        canonical = rule["canonical"]
        ch_suffix = rule.get("save_as_channel")
        extra_sfx = rule.get("save_as_extra")

        if rule.get("generated"):
            if ch_suffix:
                _write(beam_mask, ch_suffix)
            continue

        if ch_suffix:
            if rule.get("compute_sdm") and canonical in sdms:
                _write(sdms[canonical], ch_suffix)
            elif canonical == "Body":
                _write(body_mask, ch_suffix)
            elif canonical == "PTV":
                _write(masks[canonical], ch_suffix)
            else:
                _write(masks[canonical], ch_suffix)

        if extra_sfx:
            _write(masks[canonical], extra_sfx)

    # ---- Individual SIB PTV files (Painter's Algorithm support) --------
    ptv_rule = _STRUCT_RULES.get("PTV", {})
    sib_map  = ptv_rule.get("sib_canonical", [])
    if sib_map:
        # individual_ptv_masks: one per matched roi_name
        individual_masks = {}
        for p_name in matched.get("PTV", []):
            individual_masks[p_name] = rtstruct_contour_to_mask(rs_ds, p_name, ct_image)
        # Accumulate into canonical buckets
        accumulated = {}
        for p_name, p_mask in individual_masks.items():
            canonical_target = None
            for entry in sib_map:
                if re.match(entry["pattern"], p_name, re.IGNORECASE):
                    canonical_target = entry["maps_to"]
                    break
            if canonical_target is None:
                canonical_target = p_name.replace(" ", "_").replace("/", "_")
            if canonical_target in accumulated:
                accumulated[canonical_target] = np.maximum(
                    accumulated[canonical_target], p_mask
                )
            else:
                accumulated[canonical_target] = p_mask
        for canon_name, p_mask in accumulated.items():
            _write(p_mask, canon_name)
            print(f"         SIB: {canon_name}  ({p_mask.sum():,} voxels)")

    # ---- Dose label ----------------------------------------------------
    sitk.WriteImage(
        numpy_to_sitk(dose_array.astype(np.float32), ct_image),
        os.path.join(labels_dir, f"{case_name}.nii.gz"),
    )

    n_ch = len([r for r in _pre["structure_matching"] if r.get("save_as_channel")])
    print(f"  ✓ Done! Saved {n_ch} input channels + auxiliary masks + label for {case_name}")
    return True


# ===========================================================================
# dataset.json — channel names from config
# ===========================================================================

def create_dataset_json(output_dir, num_cases):
    dataset_json = {
        "channel_names": _CHANNEL_NAMES,
        "labels":        {"0": "dose"},
        "numTraining":   num_cases,
        "file_ending":   ".nii.gz",
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\n  dataset.json saved to {json_path}")
    return json_path


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    images_dir = os.path.join(OUTPUT_DIR, "imagesTr")
    labels_dir = os.path.join(OUTPUT_DIR, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    patient_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    success_count = 0
    failed = []
    for i, pdir in enumerate(patient_dirs):
        try:
            ok = convert_patient(pdir, case_id=i,
                                 images_dir=images_dir, labels_dir=labels_dir)
            if ok:
                success_count += 1
            else:
                failed.append(os.path.basename(pdir)[:12])
        except Exception as e:
            print(f"\n  *** ERROR processing {os.path.basename(pdir)[:12]}: {e} ***")
            failed.append(os.path.basename(pdir)[:12])

    create_dataset_json(OUTPUT_DIR, success_count)
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Successfully converted: {success_count}/{len(patient_dirs)}")
    if failed:
        print(f"  Failed patients: {failed}")