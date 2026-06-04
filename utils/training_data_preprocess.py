"""
DICOM → nnU-Net v2 Converter for Prostate Dose Prediction
==========================================================
Converts raw DICOM data (CT + RTStruct + RTPlan + RTDose) into the nnU-Net v2
regression format.

Input channels:
  0 = CT (Hounsfield Units)
  1 = PTV binary mask (union of all target volumes)
  2 = Bladder signed distance map (mm, negative inside organ)
  3 = Anorectum signed distance map (mm, negative inside organ)
  4 = Body Mask (binary: 1 = inside patient, 0 = air outside body)
  5 = Penile Bulb binary mask (binary: 1 = inside organ, 0 = outside)
  6 = BEV Beam Frustum Mask (binary: 1 = intersected by beam frustum + 7 mm penumbra margin)

Label:
  RTDose in Gy (continuous values for regression)
"""

import os
import re
import json
import glob
import yaml
from dotenv import load_dotenv
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

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "config.env")
load_dotenv(_ENV_PATH)

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
DATASET_NAME = os.environ["DATASET_NAME"]

# Load site config from YAML
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", os.environ["SITE_CONFIG_YAML"])
with open(_CONFIG_PATH, "r") as _f:
    SITE_CONFIG = yaml.safe_load(_f)


# Build STRUCTURE_PATTERNS from YAML channels, auxiliary_masks, and ptv_levels
STRUCTURE_PATTERNS = {
    "PTV": [rf"^{name}$" for name in SITE_CONFIG["ptv_levels"]],
}
# Collect all unique structure names referenced in channels and auxiliary_masks
for _ch in SITE_CONFIG["channels"]:
    _struct = _ch.get("structure")
    if _struct:
        STRUCTURE_PATTERNS[_struct] = [rf"^{_struct}$"]
for _aux in SITE_CONFIG.get("auxiliary_masks", []):
    if "structure" in _aux:
        STRUCTURE_PATTERNS[_aux["structure"]] = [rf"^{_aux['structure']}$"]
    for _s in _aux.get("structures", []):
        STRUCTURE_PATTERNS[_s] = [rf"^{_s}$"]

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def find_dicom_subdir(patient_dir):
    subdirs = [d for d in os.listdir(patient_dir)
               if os.path.isdir(os.path.join(patient_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(patient_dir, subdirs[0])
    return patient_dir

def sort_dicom_files(dicom_dir):
    """
    Safely opens all files in a directory and sorts them by DICOM Modality.
    This ignores filenames completely and relies on the DICOM header.
    """
    sorted_files = {
        "RTSTRUCT": [],
        "RTPLAN": [],
        "RTDOSE": []
    }
    
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue
            
        try:
            # stop_before_pixels=True makes this extremely fast
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if hasattr(ds, 'Modality') and ds.Modality in sorted_files:
                sorted_files[ds.Modality].append(fpath)
        except Exception:
            # Skip files that aren't valid DICOMs
            pass
            
    return sorted_files

def find_correct_rtstruct(plan_files, struct_files):
    if not struct_files:
        raise FileNotFoundError("No RTStruct files found in directory!")
        
    if len(struct_files) == 1:
        return struct_files[0]
        
    if not plan_files:
        print("    WARNING: Multiple RTStructs, but no RTPlan found. Using first RTStruct.")
        return struct_files[0]

    plan_ds = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
    ref_uid = None
    if hasattr(plan_ds, 'ReferencedStructureSetSequence'):
        ref_uid = plan_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

    for sf in struct_files:
        ds = pydicom.dcmread(sf, stop_before_pixels=True)
        if ds.SOPInstanceUID == ref_uid:
            return sf

    print("    WARNING: No RTStruct matched RTPlan reference, using first one")
    return struct_files[0]

def match_structure_name(roi_names, structure_type):
    patterns = STRUCTURE_PATTERNS[structure_type]
    for pattern in patterns:
        for name in roi_names:
            if re.match(pattern, name, re.IGNORECASE):
                return name
    return None

def match_all_structure_names(roi_names, structure_type):
    """
    Returns ALL roi_names that match any pattern for the given structure_type.
    Used for merging dual PTVs (e.g., prostate PTV_62 + pelvic node CTV_44)
    into a single union mask for ch_1.
    """
    patterns = STRUCTURE_PATTERNS[structure_type]
    matched = []
    for name in roi_names:
        for pattern in patterns:
            if re.match(pattern, name, re.IGNORECASE):
                matched.append(name)
                break  # avoid double-counting the same name
    return matched

def rtstruct_all_contours_to_mask(rs_ds, roi_names_list, ct_image):
    """
    Union-rasterize all contours in roi_names_list into a single binary mask.
    Any voxel inside ANY of the named structures is set to 1.
    This correctly handles patients with dual PTVs.
    """
    ct_array = sitk.GetArrayFromImage(ct_image)
    union_mask = np.zeros(ct_array.shape, dtype=np.uint8)
    for roi_name in roi_names_list:
        single = rtstruct_contour_to_mask(rs_ds, roi_name, ct_image)
        union_mask = np.maximum(union_mask, single)  # element-wise OR
    return union_mask

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

    if contour_seq is None or not hasattr(contour_seq, 'ContourSequence'):
        return mask

    for contour in contour_seq.ContourSequence:
        points = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        pixel_coords = []
        for pt in points:
            idx = ct_image.TransformPhysicalPointToContinuousIndex((float(pt[0]), float(pt[1]), float(pt[2])))
            pixel_coords.append(idx)

        pixel_coords = np.array(pixel_coords)
        slice_idx = int(round(pixel_coords[0, 2]))

        if 0 <= slice_idx < mask.shape[0]:
            rows = pixel_coords[:, 1]
            cols = pixel_coords[:, 0]
            rr, cc = polygon(rows, cols, shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1

    return mask

def compute_signed_distance_map(binary_mask, spacing_mm):
    if binary_mask.sum() == 0:
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0
    dist_outside = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)
    dist_inside = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)
    return (dist_outside - dist_inside).astype(np.float32)

def load_rtdose_as_sitk(dose_files, ct_image):
    if not dose_files:
        raise FileNotFoundError("No RTDose file found!")

    ds = pydicom.dcmread(dose_files[0])
    dose_array = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    dose_image = sitk.GetImageFromArray(dose_array)

    dose_origin = [float(x) for x in ds.ImagePositionPatient]
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
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetOrigin(reference_image.GetOrigin())
    sitk_image.SetSpacing(reference_image.GetSpacing())
    sitk_image.SetDirection(reference_image.GetDirection())
    return sitk_image


# ===========================================================================
# BEV FRUSTUM BEAM MASK GENERATION (Channel 6)
# ===========================================================================

DEFAULT_GANTRY_ANGLES = [0, 51, 102, 154, 205, 257, 308]  # 7 equispaced angles (fallback)
SAD_MM = 1000.0          # Standard Source-to-Axis Distance in mm
PENUMBRA_MM = 7.0        # Uniform isotropic penumbra margin in mm

def _parse_gantry_angles(plan_files):
    """
    Extract unique macro gantry angles from the first control point of each
    treatment beam in an RTPlan DICOM.
    Falls back to DEFAULT_GANTRY_ANGLES if no plan or no angles are found.
    """
    if not plan_files:
        print("         WARNING: No RTPlan found. Defaulting to 7 equispaced gantry angles.")
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
            print("         WARNING: No gantry angles in RTPlan. Defaulting to 7 equispaced angles.")
            return DEFAULT_GANTRY_ANGLES
        unique_angles = sorted(set(angles))
        print(f"         Gantry Angles from RTPlan: {unique_angles}")
        return unique_angles
    except Exception as e:
        print(f"         WARNING: Could not read RTPlan ({e}). Defaulting to 7 equispaced angles.")
        return DEFAULT_GANTRY_ANGLES


def generate_bev_beam_mask(plan_files, ct_image, ptv_mask_array):
    """
    Build a 3-D binary BEV frustum mask.

    For each gantry angle:
      1. Place a virtual point source at SAD_MM distance along the beam direction.
      2. Ray-cast a divergent frustum from the source through every PTV voxel.
      3. Expand by PENUMBRA_MM (uniform isotropic margin).

    Returns a binary float32 numpy array (ZYX, same shape as CT).
    """
    gantry_angles = _parse_gantry_angles(plan_files)

    shape_zyx = sitk.GetArrayViewFromImage(ct_image).shape
    origin   = np.array(ct_image.GetOrigin())             # (x, y, z)
    spacing  = np.array(ct_image.GetSpacing())            # (sx, sy, sz)
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    # Build physical-coordinate grid (ZYX → XYZ vectorised)
    zi, yi, xi = np.meshgrid(
        np.arange(shape_zyx[0]),
        np.arange(shape_zyx[1]),
        np.arange(shape_zyx[2]),
        indexing='ij'
    )  # all shape (Z, Y, X)
    voxel_indices = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)  # (N, 3) XYZ
    phys_pts = origin + (voxel_indices * spacing) @ direction.T              # (N, 3) in mm

    # PTV centre of mass in physical space (isocenter)
    ptv_z, ptv_y, ptv_x = np.where(ptv_mask_array > 0.5)
    if len(ptv_z) == 0:
        print("         WARNING: PTV mask empty — returning blank beam mask.")
        return np.zeros(shape_zyx, dtype=np.float32)

    iso_idx = np.array([
        ptv_x.mean(),
        ptv_y.mean(),
        ptv_z.mean(),
    ])  # XYZ voxel indices
    iso_phys = origin + (iso_idx * spacing) @ direction.T   # mm, shape (3,)
    print(f"         PTV isocenter (mm): {np.round(iso_phys, 1)}")

    beam_flat = np.zeros(len(phys_pts), dtype=np.float32)

    for angle_deg in gantry_angles:
        theta = np.deg2rad(angle_deg)
        # IEC 61217 gantry: beam arrives from +Y rotated by theta
        #   beam_dir points FROM source TOWARD isocenter (i.e. beam travel direction)
        beam_dir = np.array([np.sin(theta), -np.cos(theta), 0.0])  # unit vector (XYZ)
        source_pos = iso_phys - beam_dir * SAD_MM                   # virtual point source

        # Vector from source to every voxel
        vec = phys_pts - source_pos                         # (N, 3)

        # Project onto beam axis → denominator for divergence
        depth = vec @ beam_dir                              # (N,)

        # Perpendicular distance in the plane transverse to beam
        proj_along = depth[:, np.newaxis] * beam_dir       # (N, 3)
        perp_vec   = vec - proj_along                       # (N, 3)
        perp_dist  = np.linalg.norm(perp_vec, axis=1)      # (N,) in mm

        # Frustum radius at each voxel: grows linearly with depth from source
        # PTV half-width at isocenter (SAD from source) → sets the opening angle
        ptv_indices = np.stack([ptv_x, ptv_y, ptv_z], axis=1)  # (M, 3) XYZ
        ptv_phys = origin + (ptv_indices * spacing) @ direction.T  # (M, 3) physical coords
        vec_ptv = ptv_phys - source_pos                    # (M, 3)
        depth_ptv = vec_ptv @ beam_dir                     # (M,)
        perp_ptv  = np.linalg.norm(vec_ptv - depth_ptv[:, np.newaxis] * beam_dir, axis=1)

        # Frustum half-angle (half-angle in radians)
        if depth_ptv.max() > 0:
            half_angle = np.arctan2(perp_ptv.max(), depth_ptv[depth_ptv > 0].min())
        else:
            half_angle = np.deg2rad(5.0)  # fallback 5°

        # Radius at each voxel depth (divergent frustum) + penumbra
        radius_at_voxel = np.abs(depth) * np.tan(half_angle) + PENUMBRA_MM

        # Mark voxels inside the frustum
        inside = (depth > 0) & (perp_dist <= radius_at_voxel)
        beam_flat[inside] = 1.0

    beam_zyx = beam_flat.reshape(shape_zyx[0], shape_zyx[1], shape_zyx[2])
    n_voxels = int(beam_zyx.sum())
    print(f"         BEV Beam Mask Voxels (all {len(gantry_angles)} beams): {n_voxels:,}")
    return beam_zyx


# ===========================================================================
# MAIN CONVERTER LOOP
# ===========================================================================

def convert_patient(patient_dir, case_id, images_dir, labels_dir):
    cfg = SITE_CONFIG
    folder_uuid = os.path.basename(patient_dir)
    pid = folder_uuid.replace('.', '_')
    case_name = f"prostate_{pid}"
    print(f"\n{'='*60}")
    print(f"  Converting: {folder_uuid}  →  {case_name}")
    print(f"{'='*60}")

    dicom_dir = find_dicom_subdir(patient_dir)
    dicom_files = sort_dicom_files(dicom_dir)
    plan_files = dicom_files.get("RTPLAN", [])
    struct_files = dicom_files.get("RTSTRUCT", [])
    dose_files = dicom_files.get("RTDOSE", [])

    # ── Step 1: Load CT ──────────────────────────────────────────────
    print("  [1] Loading CT volume...")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print("  *** SKIPPING: No DICOM series found ***")
        return False
    best_series = max(
        (reader.GetGDCMSeriesFileNames(dicom_dir, sid) for sid in series_ids),
        key=len,
    )
    reader.SetFileNames(best_series)
    ct_image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing = ct_image.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    print(f"         Shape: {ct_array.shape}, Spacing: {spacing}")
    if ct_image.GetDimension() != 3:
        print(f"  *** SKIPPING: CT is {ct_image.GetDimension()}D ***")
        return False

    # ── Step 2: Parse RTStruct ───────────────────────────────────────
    print("  [2] Parsing RTStruct contours...")
    rs_file = find_correct_rtstruct(plan_files, struct_files)
    rs_ds = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]

    # PTV union mask (needed by several channel types)
    ptv_names = match_all_structure_names(roi_names, "PTV")
    if not ptv_names:
        print("  *** SKIPPING: No PTV/CTV structures found ***")
        return False
    print(f"         PTV structures ({len(ptv_names)}): {ptv_names}")
    ptv_mask = rtstruct_all_contours_to_mask(rs_ds, ptv_names, ct_image)
    if ptv_mask.sum() == 0:
        print("  *** SKIPPING: PTV mask is empty ***")
        return False

    # ── Step 3: Process channels from YAML ───────────────────────────
    print("  [3] Processing input channels...")
    channel_arrays = []
    for idx, ch in enumerate(cfg["channels"]):
        ch_type = ch["type"]
        ch_name = ch["name"]
        struct  = ch.get("structure")
        optional = ch.get("optional", True)

        if ch_type == "ct":
            arr = ct_array.astype(np.float32)

        elif ch_type == "ptv_union":
            arr = ptv_mask.astype(np.float32)

        elif ch_type == "sdm":
            roi = match_structure_name(roi_names, struct)
            if roi is None and not optional:
                print(f"  *** SKIPPING: Required structure {struct} not found ***")
                return False
            if roi:
                mask = rtstruct_contour_to_mask(rs_ds, roi, ct_image)
            else:
                mask = np.zeros_like(ct_array, dtype=np.uint8)
            arr = compute_signed_distance_map(mask, spacing_zyx)
            print(f"         ch_{idx} ({ch_name}): {roi or 'EMPTY'}")

        elif ch_type == "binary_mask":
            roi = match_structure_name(roi_names, struct)
            if roi is None and not optional:
                print(f"  *** SKIPPING: Required structure {struct} not found ***")
                return False
            if roi:
                arr = rtstruct_contour_to_mask(rs_ds, roi, ct_image).astype(np.float32)
                print(f"         ch_{idx} ({ch_name}): {roi}  ({int(arr.sum()):,} voxels)")
            else:
                arr = np.zeros(ct_array.shape, dtype=np.float32)
                print(f"         ch_{idx} ({ch_name}): NOT FOUND — empty mask")

        elif ch_type == "body_mask":
            roi = match_structure_name(roi_names, struct) if struct else None
            if roi:
                arr = rtstruct_contour_to_mask(rs_ds, roi, ct_image).astype(np.float32)
                print(f"         ch_{idx} ({ch_name}): {roi}  ({int(arr.sum()):,} voxels)")
            else:
                arr = (ct_array > -300.0).astype(np.float32)
                print(f"         ch_{idx} ({ch_name}): HU threshold fallback  ({int(arr.sum()):,} voxels)")

        elif ch_type == "bev_beam":
            arr = generate_bev_beam_mask(plan_files, ct_image, ptv_mask)

        else:
            raise ValueError(f"Unknown channel type '{ch_type}' for ch_{idx} ({ch_name})")

        channel_arrays.append(arr)

    # ── Step 4: Auxiliary masks (loss-only) ──────────────────────────
    print("  [4] Processing auxiliary masks...")
    aux_masks = {}
    for aux in cfg.get("auxiliary_masks", []):
        tag = aux["tag"]
        optional = aux.get("optional", True)
        # single structure
        if "structure" in aux:
            roi = match_structure_name(roi_names, aux["structure"])
            if roi:
                mask = rtstruct_contour_to_mask(rs_ds, roi, ct_image)
                print(f"         {tag}: {roi}  ({mask.sum():,} voxels)")
            elif not optional:
                print(f"  *** SKIPPING: Required auxiliary {aux['structure']} not found ***")
                return False
            else:
                mask = np.zeros_like(ct_array, dtype=np.uint8)
                print(f"         {tag}: NOT FOUND — empty mask")
        # merged structures (e.g. Femur L+R)
        elif "structures" in aux:
            mask = np.zeros_like(ct_array, dtype=np.uint8)
            found = []
            for s in aux["structures"]:
                roi = match_structure_name(roi_names, s)
                if roi:
                    mask = np.maximum(mask, rtstruct_contour_to_mask(rs_ds, roi, ct_image))
                    found.append(roi)
            print(f"         {tag}: merged {found or 'NONE'}  ({mask.sum():,} voxels)")
        else:
            mask = np.zeros_like(ct_array, dtype=np.uint8)
        aux_masks[tag] = mask

    # ── Step 5: Individual PTV masks for SIB ─────────────────────────
    print("  [5] Rasterizing individual PTVs for SIB mapping...")
    individual_ptv_masks = {}
    for p_name in ptv_names:
        individual_ptv_masks[p_name] = rtstruct_contour_to_mask(rs_ds, p_name, ct_image)

    # Map each matched PTV to its canonical dose level from YAML ptv_levels
    ptv_levels = cfg.get("ptv_levels", [])
    accumulated_ptvs = {}
    for p_name, p_mask in individual_ptv_masks.items():
        canonical = None
        for level in ptv_levels:
            dose_num = level.split("_")[-1]   # "PTV_60" → "60"
            if re.search(dose_num, p_name, re.IGNORECASE):
                canonical = level.replace("_", "")  # "PTV_60" → "PTV60"
                break
        if canonical is None:
            canonical = p_name.replace(" ", "_").replace("/", "_")
        if canonical in accumulated_ptvs:
            accumulated_ptvs[canonical] = np.maximum(accumulated_ptvs[canonical], p_mask)
        else:
            accumulated_ptvs[canonical] = p_mask

    # ── Step 6: Load RTDose label ────────────────────────────────────
    print("  [6] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dose_files, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)

    # ── Step 7: Save NIfTI files ─────────────────────────────────────
    print("  [7] Saving NIfTI files...")
    n_channels = len(channel_arrays)

    for idx, arr in enumerate(channel_arrays):
        fpath = os.path.join(images_dir, f"{case_name}_{idx:04d}.nii.gz")
        sitk.WriteImage(numpy_to_sitk(arr.astype(np.float32), ct_image), fpath)

    for tag, mask in aux_masks.items():
        fpath = os.path.join(images_dir, f"{case_name}_{tag}.nii.gz")
        sitk.WriteImage(numpy_to_sitk(mask.astype(np.float32), ct_image), fpath)

    for canonical, p_mask in accumulated_ptvs.items():
        fpath = os.path.join(images_dir, f"{case_name}_{canonical}.nii.gz")
        sitk.WriteImage(numpy_to_sitk(p_mask.astype(np.float32), ct_image), fpath)

    sitk.WriteImage(numpy_to_sitk(dose_array.astype(np.float32), ct_image),
                    os.path.join(labels_dir, f"{case_name}.nii.gz"))

    n_aux = len(aux_masks)
    n_sib = len(accumulated_ptvs)
    print(f"  ✓ Done! Saved {n_channels} channels + {n_aux} aux masks + {n_sib} SIB masks + 1 label")
    return True

def create_dataset_json(output_dir, num_cases):
    channel_names = {
        str(i): ch["name"] for i, ch in enumerate(SITE_CONFIG["channels"])
    }
    label_cfg = SITE_CONFIG.get("label", {"name": "dose"})
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"0": label_cfg["name"]},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\n  dataset.json saved to {json_path}")
    return json_path

if __name__ == "__main__":
    images_dir = os.path.join(OUTPUT_DIR, "imagesTr")
    labels_dir = os.path.join(OUTPUT_DIR, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    patient_dirs = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    success_count = 0
    failed = []
    for i, pdir in enumerate(patient_dirs):
        try:
            ok = convert_patient(pdir, case_id=i, images_dir=images_dir, labels_dir=labels_dir)
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
