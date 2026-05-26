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

# DATA_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/data/Prostate PRIME Standard arm d69"
# OUTPUT_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/nnUNet_raw/Dataset001_ProstateDose"
# DATASET_NAME = "Dataset001_ProstateDose"

DATA_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/dicom"
OUTPUT_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/raw_dicom_nifti"
DATASET_NAME = "Dataset001_ProstateDose_Test"

# Structure name matching patterns (case-insensitive, priority order)
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTVP$", r"^CTV_62", r"^PTV_62", r"^CTV62$", r"^PTV62$", 
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
    # Auxiliary OARs — used ONLY in the loss function, not as model input channels.
    # Saved as separate NIfTI files (_bowel.nii.gz, _femur.nii.gz).
    # Missing in any patient → empty mask (no skip).
    "Bag_Bowel": [
        r"^Bag_?Bowel$", r"^Bag_?Bowel\s+NOS.*", r"^BagBowel.*",
    ],
    "Femur_L": [
        r"^Femur_?Head_?L.*", r"^L_?Femur.*", r"^Left_?Femur.*",
    ],
    "Femur_R": [
        r"^Femur_?Head_?R.*", r"^R_?Femur.*", r"^Right_?Femur.*",
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
# IMRT BEAM PRIOR GENERATION (Channel 5)
# ===========================================================================

def generate_beam_mask(plan_files, ct_image, ptv_mask):
    if not plan_files:
        print("         WARNING: No RTPlan found for beam mask generation. Outputting empty mask.")
        return np.zeros(sitk.GetArrayViewFromImage(ct_image).shape, dtype=np.float32)

    plan = pydicom.dcmread(plan_files[0], stop_before_pixels=True)

    isocenter_mm = None
    gantry_angles_deg = []

    if hasattr(plan, 'BeamSequence'):
        for beam in plan.BeamSequence:
            b_type = getattr(beam, 'BeamType', '')
            d_type = getattr(beam, 'TreatmentDeliveryType', '')
            if b_type == "STATIC" or d_type == "TREATMENT":
                if hasattr(beam, 'ControlPointSequence') and len(beam.ControlPointSequence) > 0:
                    cp0 = beam.ControlPointSequence[0]
                    if isocenter_mm is None and hasattr(cp0, 'IsocenterPosition'):
                        isocenter_mm = np.array(cp0.IsocenterPosition)
                    if hasattr(cp0, 'GantryAngle'):
                        gantry_angles_deg.append(float(cp0.GantryAngle))

    if isocenter_mm is None or not gantry_angles_deg:
        print("         WARNING: Could not extract Isocenter or Gantry Angles. Outputting empty mask.")
        return np.zeros(sitk.GetArrayViewFromImage(ct_image).shape, dtype=np.float32)

    print(f"         Isocenter (mm): {isocenter_mm}")
    print(f"         Gantry Angles: {gantry_angles_deg}")

    ptv_z, ptv_y, ptv_x = np.where(ptv_mask > 0.5)
    ptv_physical_points = []
    for x, y, z in zip(ptv_x, ptv_y, ptv_z):
        ptv_physical_points.append(ct_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
    
    if len(ptv_physical_points) > 0:
        ptv_physical_points = np.array(ptv_physical_points)
        distances = np.linalg.norm(ptv_physical_points - isocenter_mm, axis=1)
        cylinder_radius_mm = np.max(distances) + 10.0
        print(f"         Dynamic Beam Radius (mm): {cylinder_radius_mm:.2f}")
    else:
        cylinder_radius_mm = 50.0
        print(f"         WARNING: Empty PTV, defaulting Beam Radius to {cylinder_radius_mm} mm")

    # Vectorized SimpleITK physical mapping
    shape_zyx = sitk.GetArrayViewFromImage(ct_image).shape
    shape_xyz = (shape_zyx[2], shape_zyx[1], shape_zyx[0])

    spacing = np.array(ct_image.GetSpacing())
    origin = np.array(ct_image.GetOrigin())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    x_idx = np.arange(shape_xyz[0])
    y_idx = np.arange(shape_xyz[1])
    z_idx = np.arange(shape_xyz[2])
    X_idx, Y_idx, Z_idx = np.meshgrid(x_idx, y_idx, z_idx, indexing='ij')

    indices = np.stack([X_idx.ravel(), Y_idx.ravel(), Z_idx.ravel()], axis=1)
    scaled_indices = indices * spacing
    physical_points = origin + np.dot(scaled_indices, direction.T)

    beam_mask_flat = np.zeros(len(physical_points), dtype=np.float32)

    for angle in gantry_angles_deg:
        theta_rad = np.deg2rad(angle)
        # IEC 61217 to Cartesian direction mapping
        beam_dir = np.array([np.sin(theta_rad), -np.cos(theta_rad), 0.0])
        beam_dir = beam_dir / np.linalg.norm(beam_dir)

        vec_to_iso = physical_points - isocenter_mm
        projection_length = np.sum(vec_to_iso * beam_dir, axis=1)
        projection_vec = projection_length[:, np.newaxis] * beam_dir
        perp_distance = np.linalg.norm(vec_to_iso - projection_vec, axis=1)

        beam_mask_flat[perp_distance <= cylinder_radius_mm] = 1.0

    beam_mask_xyz = beam_mask_flat.reshape(shape_xyz[0], shape_xyz[1], shape_xyz[2])
    return np.transpose(beam_mask_xyz, (2, 1, 0)) # Return as ZYX for SimpleITK


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

    print("  [1/7] Loading CT volume...")
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

    print("  [2/7] Parsing RTStruct contours...")
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

    print("  [3/7] Rasterizing contour masks (union of all PTVs)...")
    ptv_mask = rtstruct_all_contours_to_mask(rs_ds, ptv_names, ct_image)
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

    if ptv_mask.sum() == 0:
        print(f"  *** SKIPPING: PTV mask is empty ***")
        return False

    print("  [4/7] Computing signed distance maps...")
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = compute_signed_distance_map(bladder_mask, spacing_zyx)
    anorectum_sdm = compute_signed_distance_map(anorectum_mask, spacing_zyx)

    print("  [5/7] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dose_files, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)

    print("  [6/8] Generating IMRT Beam Prior (Channel 5)...")
    beam_mask = generate_beam_mask(plan_files, ct_image, ptv_mask)
    print(f"         Beam Prior Voxels: {beam_mask.sum():,}")

    print("  [7/8] Computing Body Mask from CT (Channel 6)...")
    # Body mask: any voxel with HU > -300 is considered inside the patient.
    # This threshold reliably separates external air (-1000 HU) from soft tissue.
    # The model uses this channel to learn that dose outside the body is always 0.
    BODY_HU_THRESHOLD = -300.0
    body_mask_array = (ct_array > BODY_HU_THRESHOLD).astype(np.float32)
    print(f"         Body Mask Voxels: {int(body_mask_array.sum()):,}")

    print("  [8/8] Saving NIfTI files...")
    case_name = f"prostate_{case_id:03d}"

    sitk.WriteImage(numpy_to_sitk(ct_array.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0000.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(ptv_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0001.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(bladder_sdm, ct_image),
                    os.path.join(images_dir, f"{case_name}_0002.nii.gz"))
    
    sitk.WriteImage(numpy_to_sitk(anorectum_sdm, ct_image),
                    os.path.join(images_dir, f"{case_name}_0003.nii.gz"))

    # Channel 4: IMRT Beam Prior
    sitk.WriteImage(numpy_to_sitk(beam_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_0004.nii.gz"))

    # Channel 5: Body Mask
    sitk.WriteImage(numpy_to_sitk(body_mask_array, ct_image),
                    os.path.join(images_dir, f"{case_name}_0005.nii.gz"))

    # Auxiliary OAR masks (loss-only, NOT model input channels).
    # Bag_Bowel: V45Gy < 30% constraint in loss function.
    # Femur (merged L+R): V50Gy < 5% constraint in loss function.
    sitk.WriteImage(numpy_to_sitk(bowel_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_bowel.nii.gz"))
    sitk.WriteImage(numpy_to_sitk(femur_mask.astype(np.float32), ct_image),
                    os.path.join(images_dir, f"{case_name}_femur.nii.gz"))

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
            "4": "IMRT_Beam_Prior",
            "5": "Body_Mask"
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
