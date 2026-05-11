"""
DICOM → nnU-Net v2 Converter for Prostate Dose Prediction
==========================================================
Converts raw DICOM data (CT + RTStruct + RTPlan + RTDose) into the nnU-Net v2
regression format.

Input channels:
  0 = CT (Hounsfield Units)
  1 = PTV binary mask (highest-dose target)
  2 = Bladder signed distance map (mm, negative inside organ)
  3 = Anorectum signed distance map (mm, negative inside organ)

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
# pyrefly: ignore [missing-import]
from skimage.draw import polygon

# CONFIGURATION

DATA_DIR = os.path.join(os.getcwd(), "Prostate prime d11 CT RT RP and RD")
OUTPUT_DIR = os.path.join(os.getcwd(), "nnUNet_raw/Dataset001_ProstateDose")
DATASET_NAME = "Dataset001_ProstateDose"

# Structure name matching patterns (case-insensitive, priority order)
# We try each pattern in order and use the FIRST match found.
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTVP$",           # Simple naming (CTVP)
        r"^CTV_62",          # CTV_62/20
        r"^PTV_62",          # PTV_62/20
        r"^CTV62$",          # CTV62
        r"^PTV62$",          # PTV62
        r"^CTV_36",          # SBRT: CTV_36.25/5
        r"^PTV_36",          # SBRT: PTV_36.25/5
        r"^CTV 36",          # SBRT: CTV 36.25 (space variant)
        r"^PTV 36",          # SBRT: PTV 36.25 (space variant)
        r"^CTV_44",          # Lower-dose CTV
        r"^PTV_44",          # Lower-dose PTV
        r"^CTV_25",          # SBRT: CTV_25/5
        r"^PTV_25",          # SBRT: PTV_25/5
        r"^CTV 25",          # SBRT: CTV 25 (space variant)
        r"^PTV 25",          # SBRT: PTV 25 (space variant)
    ],
    "Bladder": [
        r"^Bladder$",        # Standard
        r"^BLADDER$",        # Uppercase variant
    ],
    "Anorectum": [
        r"^Anorectum$",      # Standard
        r"^ANORECTUM$",      # Uppercase variant
        r"^Rectum$",         # Alternative name
    ],
}


def find_dicom_subdir(patient_dir):
    """
    Patient folders contain a study-UID subfolder with the actual DICOMs.
    This function finds that subfolder.
    """
    subdirs = [d for d in os.listdir(patient_dir)
               if os.path.isdir(os.path.join(patient_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(patient_dir, subdirs[0])
    # If multiple subdirs or none, return the patient dir itself
    return patient_dir


def find_correct_rtstruct(dicom_dir):
    """
    When multiple RTStruct files exist, pick the one referenced by the RTPlan.
    Returns the path to the correct RTStruct file.
    """
    plan_files = glob.glob(os.path.join(dicom_dir, "*RTPLAN*"))
    struct_files = glob.glob(os.path.join(dicom_dir, "*RTSTRUCT*"))

    if len(struct_files) == 1:
        return struct_files[0]

    if not plan_files:
        print("    WARNING: No RTPlan found, using first RTStruct")
        return struct_files[0]

    # Read the RTPlan to find the referenced RTStruct UID
    plan_ds = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
    ref_uid = None
    if hasattr(plan_ds, 'ReferencedStructureSetSequence'):
        ref_uid = plan_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

    # Match against available RTStruct files
    for sf in struct_files:
        ds = pydicom.dcmread(sf, stop_before_pixels=True)
        if ds.SOPInstanceUID == ref_uid:
            return sf

    print("    WARNING: No RTStruct matched RTPlan reference, using first one")
    return struct_files[0]


def match_structure_name(roi_names, structure_type):
    """
    Given a list of ROI names and a structure type (PTV, Bladder, Anorectum),
    find the best matching ROI name using the priority-ordered regex patterns.
    Returns the matched ROI name, or None if no match.
    """
    patterns = STRUCTURE_PATTERNS[structure_type]
    for pattern in patterns:
        for name in roi_names:
            if re.match(pattern, name, re.IGNORECASE):
                return name
    return None


def rtstruct_contour_to_mask(rs_ds, roi_name, ct_image):
    """
    Convert an RTStruct contour (identified by roi_name) into a 3D binary mask
    aligned to the CT image grid.

    How it works:
    1. RTStruct stores organ boundaries as sequences of (x,y,z) points (contours)
       on each CT slice.
    2. We convert these physical coordinates to pixel indices on the CT grid.
    3. We fill each polygon to create a binary mask slice-by-slice.
    """
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask = np.zeros(ct_array.shape, dtype=np.uint8)

    # Find the ROI number for this name
    roi_number = None
    for roi in rs_ds.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber
            break

    if roi_number is None:
        print(f"    WARNING: ROI '{roi_name}' not found in StructureSetROISequence")
        return mask

    # Find the contour data for this ROI
    contour_seq = None
    for roi_contour in rs_ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            contour_seq = roi_contour
            break

    if contour_seq is None or not hasattr(contour_seq, 'ContourSequence'):
        print(f"    WARNING: No contour data for ROI '{roi_name}'")
        return mask

    # Process each contour (one per CT slice)
    for contour in contour_seq.ContourSequence:
        # Get the 3D points: [x1, y1, z1, x2, y2, z2, ...]
        points = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)

        # Convert physical (x, y, z) to continuous pixel indices
        # SimpleITK: index = image.TransformPhysicalPointToContinuousIndex((x, y, z))
        pixel_coords = []
        for pt in points:
            idx = ct_image.TransformPhysicalPointToContinuousIndex(
                (float(pt[0]), float(pt[1]), float(pt[2]))
            )
            pixel_coords.append(idx)

        pixel_coords = np.array(pixel_coords)

        # The Z index tells us which slice this contour belongs to
        slice_idx = int(round(pixel_coords[0, 2]))

        if 0 <= slice_idx < mask.shape[0]:
            # Fill the polygon on this slice
            # skimage.draw.polygon expects (row, col) = (Y_index, X_index)
            rows = pixel_coords[:, 1]
            cols = pixel_coords[:, 0]
            rr, cc = polygon(rows, cols, shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1

    return mask


def compute_signed_distance_map(binary_mask, spacing_mm):
    """
    Compute a signed distance map from a binary mask.

    - Negative values = INSIDE the organ (danger zone)
    - Positive values = OUTSIDE the organ (safer)
    - Units: millimeters

    The model learns: "the more negative, the deeper inside the organ;
    the more positive, the further away and safer."
    """
    if binary_mask.sum() == 0:
        # If no mask, return all positive (far from any organ)
        print("    WARNING: Empty mask, returning large positive distance")
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0

    # Distance from outside to the nearest organ surface
    dist_outside = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)

    # Distance from inside to the nearest organ surface
    dist_inside = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)

    # Signed: negative inside, positive outside
    signed_dist = dist_outside - dist_inside

    return signed_dist.astype(np.float32)


def load_rtdose_as_sitk(dicom_dir, ct_image):
    """
    Load the RTDose DICOM file and resample it onto the CT grid.

    RTDose grids are typically coarser than CT and may cover a different region.
    We resample to match the CT exactly so all channels are aligned.
    """
    dose_files = glob.glob(os.path.join(dicom_dir, "*RTDOSE*"))
    if not dose_files:
        raise FileNotFoundError("No RTDose file found!")

    ds = pydicom.dcmread(dose_files[0])

    # Build the dose array in Gy
    dose_array = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)

    # Construct a SimpleITK image from the dose array
    dose_image = sitk.GetImageFromArray(dose_array)

    # Set the spatial metadata from the DICOM headers
    dose_origin = [float(x) for x in ds.ImagePositionPatient]
    dose_spacing = [float(ds.PixelSpacing[1]),  # column spacing = X
                    float(ds.PixelSpacing[0]),  # row spacing = Y
                    float(ds.GridFrameOffsetVector[1]) - float(ds.GridFrameOffsetVector[0])]  # Z spacing
    dose_image.SetOrigin(dose_origin)
    dose_image.SetSpacing(dose_spacing)
    dose_image.SetDirection(ct_image.GetDirection())  # Assume same orientation as CT

    # Resample dose onto the CT grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    dose_resampled = resampler.Execute(dose_image)

    return dose_resampled


def load_rtplan_prescription(dicom_dir):
    """
    Extract the prescribed dose (Gy) from the RTPlan DICOM.
    """
    plan_files = glob.glob(os.path.join(dicom_dir, "*RTPLAN*"))
    if not plan_files:
        return None

    ds = pydicom.dcmread(plan_files[0], stop_before_pixels=True)
    if hasattr(ds, 'DoseReferenceSequence'):
        for dr in ds.DoseReferenceSequence:
            rx_dose = getattr(dr, 'TargetPrescriptionDose', None)
            if rx_dose is not None:
                return float(rx_dose)
    return None


def numpy_to_sitk(array, reference_image):
    """
    Convert a NumPy array to a SimpleITK image, copying spatial metadata
    from a reference image (the CT). This ensures all channels are aligned.
    """
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetOrigin(reference_image.GetOrigin())
    sitk_image.SetSpacing(reference_image.GetSpacing())
    sitk_image.SetDirection(reference_image.GetDirection())
    return sitk_image

# MAIN CONVERTER

def convert_patient(patient_dir, case_id, images_dir, labels_dir):
    """
    Convert a single patient from DICOM to nnU-Net format.
    Returns True on success, False on failure.
    """
    pid = os.path.basename(patient_dir)[:12]
    print(f"\n{'='*60}")
    print(f"  Converting: {pid}...  (case_id: prostate_{case_id:03d})")
    print(f"{'='*60}")

    dicom_dir = find_dicom_subdir(patient_dir)

    # Step 1: Load CT
    # NOTE: Some patient folders have RTDose DICOMs that SimpleITK picks up
    # as a second series, causing a 4D read. We filter to only CT files first.
    print("  [1/6] Loading CT volume...")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        print("  *** SKIPPING: No DICOM series found in directory ***")
        return False
        
    # Pick the series with the most files (this will be the CT scan, 
    # whereas RTSTRUCT/RTPLAN/RTDOSE are single or few files)
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

    # Sanity check: CT must be 3D
    if ct_image.GetDimension() != 3:
        print(f"  *** SKIPPING: CT is {ct_image.GetDimension()}D, expected 3D ***")
        return False

    # Step 2: Find and load correct RTStruct
    print("  [2/6] Parsing RTStruct contours...")
    rs_file = find_correct_rtstruct(dicom_dir)
    rs_ds = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]
    print(f"         Total ROIs: {len(roi_names)}")

    # Match structure names
    ptv_name = match_structure_name(roi_names, "PTV")
    bladder_name = match_structure_name(roi_names, "Bladder")
    anorectum_name = match_structure_name(roi_names, "Anorectum")

    print(f"         PTV matched:       '{ptv_name}'")
    print(f"         Bladder matched:   '{bladder_name}'")
    print(f"         Anorectum matched: '{anorectum_name}'")

    if not all([ptv_name, bladder_name, anorectum_name]):
        print(f"  *** SKIPPING: Could not match all required structures ***")
        print(f"      Available: {roi_names}")
        return False

    # Step 3: Rasterize contours into binary masks 
    print("  [3/6] Rasterizing contour masks...")
    ptv_mask = rtstruct_contour_to_mask(rs_ds, ptv_name, ct_image)
    bladder_mask = rtstruct_contour_to_mask(rs_ds, bladder_name, ct_image)
    anorectum_mask = rtstruct_contour_to_mask(rs_ds, anorectum_name, ct_image)

    print(f"         PTV voxels:       {ptv_mask.sum():,}")
    print(f"         Bladder voxels:   {bladder_mask.sum():,}")
    print(f"         Anorectum voxels: {anorectum_mask.sum():,}")

    if ptv_mask.sum() == 0:
        print(f"  *** SKIPPING: PTV mask is empty ***")
        return False

    # Step 4: Compute signed distance maps for OARs
    print("  [4/6] Computing signed distance maps...")
    # Spacing in (Z, Y, X) order for numpy arrays
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = compute_signed_distance_map(bladder_mask, spacing_zyx)
    anorectum_sdm = compute_signed_distance_map(anorectum_mask, spacing_zyx)

    print(f"         Bladder SDM range:   [{bladder_sdm.min():.1f}, {bladder_sdm.max():.1f}] mm")
    print(f"         Anorectum SDM range: [{anorectum_sdm.min():.1f}, {anorectum_sdm.max():.1f}] mm")

    # Step 5: Load and resample RTDose 
    print("  [5/6] Loading and resampling RTDose...")
    dose_image = load_rtdose_as_sitk(dicom_dir, ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)
    rx_dose = load_rtplan_prescription(dicom_dir)

    print(f"         Dose range: [{dose_array.min():.2f}, {dose_array.max():.2f}] Gy")
    print(f"         Prescription: {rx_dose} Gy")

    # Step 6: Save as NIfTI 
    print("  [6/6] Saving NIfTI files...")
    case_name = f"prostate_{case_id:03d}"

    # Channel 0: CT (float32, raw HU)
    ct_f32 = ct_array.astype(np.float32)
    sitk.WriteImage(
        numpy_to_sitk(ct_f32, ct_image),
        os.path.join(images_dir, f"{case_name}_0000.nii.gz")
    )

    # Channel 1: PTV binary mask (float32, 0/1)
    sitk.WriteImage(
        numpy_to_sitk(ptv_mask.astype(np.float32), ct_image),
        os.path.join(images_dir, f"{case_name}_0001.nii.gz")
    )

    # Channel 2: Bladder signed distance map (float32, mm)
    sitk.WriteImage(
        numpy_to_sitk(bladder_sdm, ct_image),
        os.path.join(images_dir, f"{case_name}_0002.nii.gz")
    )

    # Channel 3: Anorectum signed distance map (float32, mm)
    sitk.WriteImage(
        numpy_to_sitk(anorectum_sdm, ct_image),
        os.path.join(images_dir, f"{case_name}_0003.nii.gz")
    )

    # Label: RTDose (float32, Gy)
    sitk.WriteImage(
        numpy_to_sitk(dose_array.astype(np.float32), ct_image),
        os.path.join(labels_dir, f"{case_name}.nii.gz")
    )

    print(f"  ✓ Done! Saved 4 input channels + 1 label for {case_name}")
    return True


def create_dataset_json(output_dir, num_cases):
    """
    Create the dataset.json file required by nnU-Net.

    This file tells nnU-Net:
    - How many input channels exist and what they represent
    - That this is a REGRESSION task (not segmentation)
    - The file naming convention
    """
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PTV_mask",
            "2": "Bladder_SDM",
            "3": "Anorectum_SDM",
        },
        "labels": {
            "0": "dose"
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",

        # nnU-Net v2 regression configuration:
        # "regions_class_order" is omitted for regression
        # The label is a continuous float volume (dose in Gy)
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n  dataset.json saved to {json_path}")
    return json_path


# EXECUTION

if __name__ == "__main__":
    # Create output directories
    images_dir = os.path.join(OUTPUT_DIR, "imagesTr")
    labels_dir = os.path.join(OUTPUT_DIR, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Get all patient directories
    patient_dirs = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    print(f"Found {len(patient_dirs)} patients")
    print(f"Output: {OUTPUT_DIR}")

    # Convert each patient
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
            import traceback
            traceback.print_exc()
            failed.append(os.path.basename(pdir)[:12])

    # Create dataset.json
    create_dataset_json(OUTPUT_DIR, success_count)

    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Successfully converted: {success_count}/{len(patient_dirs)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
