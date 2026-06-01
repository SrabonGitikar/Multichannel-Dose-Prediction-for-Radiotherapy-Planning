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
  4 = IMRT Beam Prior (binary mask, dynamic-radius cylinders along gantry angles)
  5 = Body Mask (binary: 1 = inside patient, 0 = air outside body)
  6 = Penile Bulb binary mask (binary: 1 = inside organ, 0 = outside)  [v3]

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

DATA_DIR = "/home/ankit/Dose_pred/Prostate prime d11 CT RT RP and RD"
OUTPUT_DIR = "/home/ankit/Dose_pred/nnUNet_raw/Dataset001_ProstateDose"
DATASET_NAME = "Dataset001_ProstateDose"

# Structure name matching patterns (case-insensitive, priority order)
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTVP$",
        r"^CTV_?60", r"^PTV_?60",          # New: PTV60 (v3 protocol)
        r"^CTV_62", r"^PTV_62", r"^CTV62$", r"^PTV62$",  # Legacy PTV62
        r"^CTV_55", r"^PTV_55", r"^CTV55$", r"^PTV55$",
        r"^CTV_54", r"^PTV_54", r"^CTV54$", r"^PTV54$",
        r"^CTV_36", r"^PTV_36", r"^CTV 36", r"^PTV 36",
        r"^CTV_44", r"^PTV_44", r"^CTV_25", r"^PTV_25",
        r"^CTV 25", r"^PTV 25",
    ],
    "Bladder": [
        r"^Bladder$", r"^BLADDER$",
    ],
    "Anorectum": [
        r"^Anorectum$", r"^ANORECTUM$", r"^Rectum$",
    ],
    # Auxiliary OARs — saved as separate NIfTI files (_bowel.nii.gz, _femur.nii.gz).
    # Missing in any patient → empty mask (no skip).
    "Bag_Bowel": [
        r"^Bag_?Bowel$", r"^Bag_?Bowel\s+NOS.*", r"^BagBowel.*",
    ],
    "Body": [
        r"^Body$", r"^EXTERNAL$", r"^Patient$", r"^Skin$",
    ],
    "Femur_L": [
        r"^Femur_?Head_?L.*", r"^L_?Femur.*", r"^Left_?Femur.*",
    ],
    "Femur_R": [
        r"^Femur_?Head_?R.*", r"^R_?Femur.*", r"^Right_?Femur.*",
    ],
    # Penile Bulb — new in v3. Saved as channel 6 (_0006.nii.gz), binary mask.
    # Missing in any patient → empty mask (no skip, loss term = 0).
    "Penile_Bulb": [
        r"^Penile_?Bulb.*",
        r"^PenileBulb.*",
        r"^Penile Bulb.*",
    ],
}

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
# MAIN CONVERTER LOOP
# ===========================================================================

def convert_patient(patient_dir, case_id, images_dir, labels_dir):
    pid = os.path.basename(patient_dir)[:12]
    print(f"\n{'='*60}")
    print(f"  Converting: {pid}...  (case_id: prostate_{case_id:03d})")
    print(f"{'='*60}")

    dicom_dir = find_dicom_subdir(patient_dir)
    
    # Sort files by DICOM Modality instead of filename
    dicom_files = sort_dicom_files(dicom_dir)
    plan_files = dicom_files.get("RTPLAN", [])
    struct_files = dicom_files.get("RTSTRUCT", [])
    dose_files = dicom_files.get("RTDOSE", [])

    print("  [1/8] Loading CT volume...")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        print("  *** SKIPPING: No DICOM series found in directory ***")
        return False
        
    best_series = None
    best_count = 0
    for sid in series_ids:
        fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(fnames) > best_count:
            best_count = len(fnames)
            best_series = fnames
            
    reader.SetFileNames(best_series)
    ct_image = reader.Execute()

    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing = ct_image.GetSpacing()
    print(f"         Shape: {ct_array.shape}, Spacing: {spacing}")

    if ct_image.GetDimension() != 3:
        print(f"  *** SKIPPING: CT is {ct_image.GetDimension()}D, expected 3D ***")
        return False

    print("  [2/8] Parsing RTStruct contours...")
    rs_file = find_correct_rtstruct(plan_files, struct_files)
    rs_ds = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]

    ptv_names   = match_all_structure_names(roi_names, "PTV")
    bladder_name  = match_structure_name(roi_names, "Bladder")
    anorectum_name = match_structure_name(roi_names, "Anorectum")

    if not ptv_names:
        print(f"  *** SKIPPING: Could not match any PTV/CTV structure ***")
        return False
    if not all([bladder_name, anorectum_name]):
        print(f"  *** SKIPPING: Could not match Bladder or Anorectum ***")
        return False
    print(f"         PTV structures found ({len(ptv_names)}): {ptv_names}")

    print("  [3/8] Rasterizing contour masks (union of all PTVs)...")
    ptv_mask = rtstruct_all_contours_to_mask(rs_ds, ptv_names, ct_image)

    print("         Rasterizing individual PTVs for SIB mapping...")
    individual_ptv_masks = {}
    for p_name in ptv_names:
        individual_ptv_masks[p_name] = rtstruct_contour_to_mask(rs_ds, p_name, ct_image)
    bladder_mask = rtstruct_contour_to_mask(rs_ds, bladder_name, ct_image)
    anorectum_mask = rtstruct_contour_to_mask(rs_ds, anorectum_name, ct_image)

    # Auxiliary OARs: Bag_Bowel and Femur heads.
    # These are optional — if missing in a patient, an empty mask is used.
    # They are NOT model input channels; saved separately for loss enforcement only.
    bowel_name   = match_structure_name(roi_names, "Bag_Bowel")
    femur_l_name = match_structure_name(roi_names, "Femur_L")
    femur_r_name = match_structure_name(roi_names, "Femur_R")

    if bowel_name:
        bowel_mask = rtstruct_contour_to_mask(rs_ds, bowel_name, ct_image)
        print(f"         Bag_Bowel: {bowel_name}  ({bowel_mask.sum():,} voxels)")
    else:
        bowel_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
        print("         Bag_Bowel: NOT FOUND — using empty mask")

    femur_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
    for fname in [femur_l_name, femur_r_name]:
        if fname:
            femur_mask = np.maximum(femur_mask,
                                    rtstruct_contour_to_mask(rs_ds, fname, ct_image))
    print(f"         Femur Heads: L={femur_l_name or 'NOT FOUND'}, "
          f"R={femur_r_name or 'NOT FOUND'}  ({femur_mask.sum():,} voxels)")

    # Penile Bulb — new v3 OAR.  Saved as model input Channel 6 (_0006.nii.gz).
    # Optional: empty mask used gracefully when absent (loss term → 0).
    penile_name = match_structure_name(roi_names, "Penile_Bulb")
    if penile_name:
        penile_bulb_mask = rtstruct_contour_to_mask(rs_ds, penile_name, ct_image)
        print(f"         Penile_Bulb: {penile_name}  ({penile_bulb_mask.sum():,} voxels)")
    else:
        penile_bulb_mask = np.zeros_like(ptv_mask, dtype=np.uint8)
        print("         Penile_Bulb: NOT FOUND — using empty mask (loss term = 0)")

    if ptv_mask.sum() == 0:
        print(f"  *** SKIPPING: PTV mask is empty ***")
        return False

    print("  [4/8] Computing signed distance maps...")
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = compute_signed_distance_map(bladder_mask, spacing_zyx)
    anorectum_sdm = compute_signed_distance_map(anorectum_mask, spacing_zyx)

    print("  [5/8] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dose_files, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)

    print("  [6/7] Computing Body Mask from RTStruct (Channel 4)...")
    body_name = match_structure_name(roi_names, "Body")
    
    if body_name:
        print(f"         Using RTStruct contour: {body_name}")
        body_mask_array = rtstruct_contour_to_mask(rs_ds, body_name, ct_image).astype(np.float32)
    else:
        print("         WARNING: No Body/External contour found. Falling back to HU threshold.")
        BODY_HU_THRESHOLD = -300.0
        body_mask_array = (ct_array > BODY_HU_THRESHOLD).astype(np.float32)
        
    print(f"         Body Mask Voxels: {int(body_mask_array.sum()):,}")

    print("\n  [Summary] Matched Structures for this Patient:")
    print(f"         PTVs: {ptv_names}")
    print(f"         Bladder: {bladder_name}")
    print(f"         Anorectum: {anorectum_name}")
    print(f"         Bag_Bowel: {bowel_name or 'NOT FOUND'}")
    print(f"         Femur Heads: L={femur_l_name or 'NOT FOUND'}, R={femur_r_name or 'NOT FOUND'}")
    print(f"         Penile_Bulb: {penile_name or 'NOT FOUND'}\n")
    print("  [7/7] Saving NIfTI files...")
    case_name = f"prostate_{case_id:03d}"

    sitk.WriteImage(numpy_to_sitk(ct_array.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0000.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(ptv_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0001.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(bladder_sdm, ct_image),
                    os.path.join(images_dir, f"{case_name}_0002.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(anorectum_sdm, ct_image),
                    os.path.join(images_dir, f"{case_name}_0003.nii.gz"))

    # Channel 4: Body Mask (Shifted from 5)
    sitk.WriteImage(numpy_to_sitk(body_mask_array.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0004.nii.gz"))

    # Channel 5: Penile Bulb binary mask (Shifted from 6)
    sitk.WriteImage(numpy_to_sitk(penile_bulb_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0005.nii.gz"))

    # Auxiliary OAR masks (loss-only, NOT model input channels).
    # Bag_Bowel: V45Gy < 90cc constraint in loss function (hardcoded threshold).
    # Femur (merged L+R): D_max < 40 Gy constraint in loss function.
    sitk.WriteImage(numpy_to_sitk(bowel_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_bowel.nii.gz"))
    sitk.WriteImage(numpy_to_sitk(femur_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_femur.nii.gz"))

    SIB_CANONICAL = {
        r"^ptv.*60|^ctv.*60|^ctvp$": "PTV60",   # v3 canonical
        r"^ptv.*62|^ctv.*62": "PTV60",            # legacy PTV62 → maps to PTV60
        r"^ptv.*55|^ctv.*55": "PTV55",
        r"^ptv.*54|^ctv.*54": "PTV54",
        r"^ptv.*44|^ctv.*44": "PTV44",
        r"^ptv.*36|^ctv.*36": "PTV36",
        r"^ptv.*25|^ctv.*25": "PTV25",
    }

    # 1. Accumulate and merge masks by canonical name to prevent overwriting
    accumulated_ptvs = {}
    for p_name, p_mask in individual_ptv_masks.items():
        canonical = None
        for pattern, cname in SIB_CANONICAL.items():
            if re.match(pattern, p_name, re.IGNORECASE):
                canonical = cname
                break
        
        if canonical is None:
            canonical = p_name.replace(" ", "_").replace("/", "_")
            
        if canonical in accumulated_ptvs:
            # Merge overlapping or adjacent volumes targeting the same dose
            accumulated_ptvs[canonical] = np.maximum(accumulated_ptvs[canonical], p_mask)
        else:
            accumulated_ptvs[canonical] = p_mask

    # 2. Write the safely accumulated canonical masks to disk
    for canonical, p_mask in accumulated_ptvs.items():
        sitk.WriteImage(numpy_to_sitk(p_mask.astype(np.float32), ct_image),
                        os.path.join(images_dir, f"{case_name}_{canonical}.nii.gz"))

    sitk.WriteImage(numpy_to_sitk(dose_array.astype(np.float32), ct_image),
                    os.path.join(labels_dir, f"{case_name}.nii.gz"))

    print(f"  ✓ Done! Saved 6 input channels + 2 auxiliary OAR masks + 1 label for {case_name}")
    return True

def create_dataset_json(output_dir, num_cases):
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PTV_mask",
            "2": "Bladder_SDM",
            "3": "Anorectum_SDM",
            "4": "Body_Mask",
            "5": "Penile_Bulb_mask",
        },
        "labels": {"0": "dose"},
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
