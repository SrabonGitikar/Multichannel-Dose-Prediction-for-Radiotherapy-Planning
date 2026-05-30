"""
DICOM → nnU-Net v2 Converter for SBRT Prostate Dose Prediction
===============================================================
Fork of dicom_to_nnunet.py for the SBRT patient cohort.

Key differences from the IMRT pipeline:
  - No IMRT Beam Prior channel (purged entirely — incompatible with SBRT)
  - No Body Mask channel
  - No Penile Bulb channel
  - No Bag_Bowel auxiliary mask
  - Femoral Heads are individual input channels (Rt and Lt separate)
  - Small Bowel is a dedicated input channel (absolute volume constraints)
  - A single PTV per patient; all PTV dose variants (36.25/30/25 Gy) map to id=1.0

Input channels (7 total):
  0 = CT (Hounsfield Units)
  1 = PTV Map (binary: all SBRT PTVs map to value 1.0)
  2 = Bladder signed distance map (mm, negative inside organ)
  3 = Anorectum signed distance map (mm, negative inside organ)
  4 = Small Bowel binary mask (V28Gy < 80cc constraint)
  5 = Right Femoral Head binary mask (V14Gy < 5% constraint)
  6 = Left Femoral Head binary mask  (V14Gy < 5% constraint)

Label:
  RTDose in Gy (continuous values for regression)
"""

import os
import re
import json
import glob
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

DATA_DIR   = os.environ.get("SBRT_DATA_DIR",   "/home/ankit/Dose_pred/SBRT_Prostate")
OUTPUT_DIR = os.environ.get("SBRT_OUTPUT_DIR", "/home/ankit/Dose_pred/nnUNet_raw/Dataset002_SBRTProstate")
DATASET_NAME = "Dataset002_SBRTProstate"

# Structure name matching patterns (case-insensitive, priority order)
STRUCTURE_PATTERNS = {
    # SBRT PTVs: 36.25 Gy (5 fr), 30 Gy (5 fr), 25 Gy (5 fr) — all → id=1.0
    "PTV": [
        r"^PTV_?36\.?25",   r"^CTV_?36\.?25",
        r"^PTV_?36",        r"^CTV_?36",
        r"^PTV_?30",        r"^CTV_?30",
        r"^PTV_?25",        r"^CTV_?25",
        r"^PTV$",           r"^CTV$",
        r"^PTV_prostate",   r"^CTV_prostate",
        r"^GTV_?prostate",  r"^GTV$",
    ],
    "Bladder": [
        r"^Bladder$", r"^BLADDER$", r"^bladder$",
    ],
    "Anorectum": [
        r"^Anorectum$", r"^ANORECTUM$", r"^Rectum$", r"^RECTUM$",
    ],
    # SBRT-specific OARs — each is a separate input channel
    "Small_Bowel": [
        r"^Small_?Bowel.*", r"^SmallBowel.*", r"^Sm_?Bowl.*",
        r"^BowelSmall.*",   r"^Bowel_Small.*",
    ],
    "Femur_R": [
        r"^Femur_?Head_?R.*", r"^R_?Femur.*", r"^Right_?Femur.*",
        r"^FemoralHead_?R.*", r"^Femoral_Head_R.*", r"^Right_Femoral.*",
    ],
    "Femur_L": [
        r"^Femur_?Head_?L.*", r"^L_?Femur.*", r"^Left_?Femur.*",
        r"^FemoralHead_?L.*", r"^Femoral_Head_L.*", r"^Left_Femoral.*",
    ],
}

# ===========================================================================
# HELPER FUNCTIONS (shared with IMRT pipeline, duplicated to keep file standalone)
# ===========================================================================

def find_dicom_subdir(patient_dir):
    subdirs = [d for d in os.listdir(patient_dir)
               if os.path.isdir(os.path.join(patient_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(patient_dir, subdirs[0])
    return patient_dir

def sort_dicom_files(dicom_dir):
    """Safely sort all files by DICOM Modality tag."""
    sorted_files = {"RTSTRUCT": [], "RTPLAN": [], "RTDOSE": []}
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if hasattr(ds, 'Modality') and ds.Modality in sorted_files:
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
        print("    WARNING: Multiple RTStructs, no RTPlan found. Using first RTStruct.")
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
    """Returns ALL roi_names matching any pattern for the given structure_type."""
    patterns = STRUCTURE_PATTERNS[structure_type]
    matched = []
    for name in roi_names:
        for pattern in patterns:
            if re.match(pattern, name, re.IGNORECASE):
                matched.append(name)
                break
    return matched

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
            idx = ct_image.TransformPhysicalPointToContinuousIndex(
                (float(pt[0]), float(pt[1]), float(pt[2]))
            )
            pixel_coords.append(idx)
        pixel_coords = np.array(pixel_coords)
        slice_idx = int(round(pixel_coords[0, 2]))
        if 0 <= slice_idx < mask.shape[0]:
            rows = pixel_coords[:, 1]
            cols = pixel_coords[:, 0]
            rr, cc = polygon(rows, cols, shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1
    return mask

def rtstruct_all_contours_to_mask(rs_ds, roi_names_list, ct_image):
    """Union-rasterize all contours in roi_names_list into a single binary mask."""
    ct_array = sitk.GetArrayFromImage(ct_image)
    union_mask = np.zeros(ct_array.shape, dtype=np.uint8)
    for roi_name in roi_names_list:
        single = rtstruct_contour_to_mask(rs_ds, roi_name, ct_image)
        union_mask = np.maximum(union_mask, single)
    return union_mask

def compute_signed_distance_map(binary_mask, spacing_mm):
    if binary_mask.sum() == 0:
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0
    dist_outside = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)
    dist_inside  = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)
    return (dist_outside - dist_inside).astype(np.float32)

def load_rtdose_as_sitk(dose_files, ct_image):
    if not dose_files:
        raise FileNotFoundError("No RTDose file found!")
    ds = pydicom.dcmread(dose_files[0])
    dose_array = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_origin  = [float(x) for x in ds.ImagePositionPatient]
    dose_spacing = [
        float(ds.PixelSpacing[1]),
        float(ds.PixelSpacing[0]),
        float(ds.GridFrameOffsetVector[1]) - float(ds.GridFrameOffsetVector[0]),
    ]
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
# MAIN CONVERTER LOOP
# ===========================================================================

def convert_patient(patient_dir, case_id, images_dir, labels_dir):
    pid = os.path.basename(patient_dir)[:16]
    print(f"\n{'='*60}")
    print(f"  Converting: {pid}...  (case_id: sbrt_{case_id:03d})")
    print(f"{'='*60}")

    dicom_dir   = find_dicom_subdir(patient_dir)
    dicom_files = sort_dicom_files(dicom_dir)
    plan_files   = dicom_files.get("RTPLAN",   [])
    struct_files = dicom_files.get("RTSTRUCT", [])
    dose_files   = dicom_files.get("RTDOSE",   [])

    # ---- [1] Load CT --------------------------------------------------
    print("  [1/6] Loading CT volume...")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print("  *** SKIPPING: No DICOM series found ***")
        return False
    best_series, best_count = None, 0
    for sid in series_ids:
        fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(fnames) > best_count:
            best_count = len(fnames)
            best_series = fnames
    reader.SetFileNames(best_series)
    ct_image  = reader.Execute()
    ct_array  = sitk.GetArrayFromImage(ct_image)
    spacing   = ct_image.GetSpacing()
    print(f"         Shape: {ct_array.shape}, Spacing: {spacing}")
    if ct_image.GetDimension() != 3:
        print(f"  *** SKIPPING: CT is {ct_image.GetDimension()}D, expected 3D ***")
        return False

    # ---- [2] Parse RTStruct -------------------------------------------
    print("  [2/6] Parsing RTStruct contours...")
    rs_file = find_correct_rtstruct(plan_files, struct_files)
    rs_ds   = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]

    ptv_names      = match_all_structure_names(roi_names, "PTV")
    bladder_name   = match_structure_name(roi_names, "Bladder")
    anorectum_name = match_structure_name(roi_names, "Anorectum")
    small_bowel_name = match_structure_name(roi_names, "Small_Bowel")
    femur_r_name   = match_structure_name(roi_names, "Femur_R")
    femur_l_name   = match_structure_name(roi_names, "Femur_L")

    if not ptv_names:
        print("  *** SKIPPING: Could not match any PTV/CTV structure ***")
        return False
    if not bladder_name or not anorectum_name:
        print("  *** SKIPPING: Could not match Bladder or Anorectum ***")
        return False
    print(f"         PTV structures found ({len(ptv_names)}): {ptv_names}")
    print(f"         Bladder: {bladder_name}")
    print(f"         Anorectum: {anorectum_name}")
    print(f"         Small_Bowel: {small_bowel_name or 'NOT FOUND — empty mask'}")
    print(f"         Femur_R: {femur_r_name or 'NOT FOUND — empty mask'}")
    print(f"         Femur_L: {femur_l_name or 'NOT FOUND — empty mask'}")

    # ---- [3] Rasterize masks ------------------------------------------
    print("  [3/6] Rasterizing contour masks...")

    # PTV union — all dose levels → single binary mask (SBRT has one target)
    ptv_mask = rtstruct_all_contours_to_mask(rs_ds, ptv_names, ct_image)
    if ptv_mask.sum() == 0:
        print("  *** SKIPPING: PTV mask is empty ***")
        return False

    bladder_mask   = rtstruct_contour_to_mask(rs_ds, bladder_name,   ct_image)
    anorectum_mask = rtstruct_contour_to_mask(rs_ds, anorectum_name, ct_image)

    # Small Bowel — optional: empty mask if missing
    if small_bowel_name:
        small_bowel_mask = rtstruct_contour_to_mask(rs_ds, small_bowel_name, ct_image)
        print(f"         Small_Bowel: {small_bowel_name}  ({small_bowel_mask.sum():,} voxels)")
    else:
        small_bowel_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
        print("         Small_Bowel: NOT FOUND — using empty mask (loss term = 0)")

    # Right Femoral Head — optional
    if femur_r_name:
        femur_r_mask = rtstruct_contour_to_mask(rs_ds, femur_r_name, ct_image)
        print(f"         Femur_R: {femur_r_name}  ({femur_r_mask.sum():,} voxels)")
    else:
        femur_r_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
        print("         Femur_R: NOT FOUND — using empty mask")

    # Left Femoral Head — optional
    if femur_l_name:
        femur_l_mask = rtstruct_contour_to_mask(rs_ds, femur_l_name, ct_image)
        print(f"         Femur_L: {femur_l_name}  ({femur_l_mask.sum():,} voxels)")
    else:
        femur_l_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
        print("         Femur_L: NOT FOUND — using empty mask")

    # ---- [4] Signed Distance Maps ------------------------------------
    print("  [4/6] Computing signed distance maps (Bladder, Anorectum)...")
    spacing_zyx   = (spacing[2], spacing[1], spacing[0])
    bladder_sdm   = compute_signed_distance_map(bladder_mask,   spacing_zyx)
    anorectum_sdm = compute_signed_distance_map(anorectum_mask, spacing_zyx)

    # ---- [5] RTDose ---------------------------------------------------
    print("  [5/6] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dose_files, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)

    # ---- [6] Save NIfTI files ----------------------------------------
    print("  [6/6] Saving NIfTI files...")
    case_name = f"sbrt_{case_id:03d}"

    # Channel 0: CT
    sitk.WriteImage(
        numpy_to_sitk(ct_array.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0000.nii.gz")
    )
    # Channel 1: PTV Map — binary float (all SBRT variants → 1.0)
    sitk.WriteImage(
        numpy_to_sitk(ptv_mask.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0001.nii.gz")
    )
    # Channel 2: Bladder SDM
    sitk.WriteImage(
        numpy_to_sitk(bladder_sdm, ct_image),
        os.path.join(images_dir, f"{case_name}_0002.nii.gz")
    )
    # Channel 3: Anorectum SDM
    sitk.WriteImage(
        numpy_to_sitk(anorectum_sdm, ct_image),
        os.path.join(images_dir, f"{case_name}_0003.nii.gz")
    )
    # Channel 4: Small Bowel binary mask
    sitk.WriteImage(
        numpy_to_sitk(small_bowel_mask.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0004.nii.gz")
    )
    # Channel 5: Right Femoral Head binary mask
    sitk.WriteImage(
        numpy_to_sitk(femur_r_mask.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0005.nii.gz")
    )
    # Channel 6: Left Femoral Head binary mask
    sitk.WriteImage(
        numpy_to_sitk(femur_l_mask.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0006.nii.gz")
    )

    # Label: RTDose in Gy
    sitk.WriteImage(
        numpy_to_sitk(dose_array.astype(np.float32), ct_image),
        os.path.join(labels_dir, f"{case_name}.nii.gz")
    )

    print(f"  ✓ Done! Saved 7 input channels + 1 label for {case_name}")
    print(f"         PTV voxels: {ptv_mask.sum():,}  |  "
          f"Bladder: {bladder_mask.sum():,}  |  "
          f"Anorectum: {anorectum_mask.sum():,}  |  "
          f"Small Bowel: {small_bowel_mask.sum():,}  |  "
          f"Femur R: {femur_r_mask.sum():,}  |  "
          f"Femur L: {femur_l_mask.sum():,}")
    return True


def create_dataset_json(output_dir, num_cases):
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PTV_Map",           # Binary; all SBRT PTV variants → 1.0
            "2": "Bladder_SDM",
            "3": "Anorectum_SDM",
            "4": "Small_Bowel_Mask",  # Binary; V28Gy < 80 cc constraint
            "5": "Femoral_Head_Rt",   # Binary; V14Gy < 5% constraint
            "6": "Femoral_Head_Lt",   # Binary; V14Gy < 5% constraint
        },
        "labels": {"0": "dose"},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "dataset": DATASET_NAME,
        "description": (
            "SBRT prostate dose prediction — 7-channel input. "
            "IMRT Beam Prior, Body Mask, and Penile Bulb purged. "
            "Femoral Heads kept as separate channels. "
            "All PTV prescriptions (36.25/30/25 Gy) map to PTV id=1.0."
        ),
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
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
            ok = convert_patient(pdir, case_id=i,
                                 images_dir=images_dir, labels_dir=labels_dir)
            if ok:
                success_count += 1
            else:
                failed.append(os.path.basename(pdir)[:16])
        except Exception as e:
            print(f"\n  *** ERROR processing {os.path.basename(pdir)[:16]}: {e} ***")
            failed.append(os.path.basename(pdir)[:16])
            import traceback
            traceback.print_exc()

    create_dataset_json(OUTPUT_DIR, success_count)
    print(f"\n{'='*60}")
    print(f"SBRT CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Successfully converted: {success_count}/{len(patient_dirs)}")
    if failed:
        print(f"  Failed ({len(failed)}): {failed}")
    print(f"  Output: {OUTPUT_DIR}")
