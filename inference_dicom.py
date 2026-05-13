"""
End-to-End DICOM Inference for Dose Prediction
==============================================
Takes a directory with DICOM CT + RTSTRUCT, preprocesses, runs inference,
and outputs a single RTDOSE DICOM file.

Input: Directory containing DICOM files (CT series + RTSTRUCT)
Output: RTDOSE DICOM file with predicted dose distribution

Usage:
    python inference_dicom.py --input-dir /path/to/dicom --output-dose predicted_dose.dcm
"""

import os
import re
import glob
import argparse
import datetime
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pydicom
# pyrefly: ignore [missing-import]
from pydicom.dataset import FileDataset
# pyrefly: ignore [missing-import]
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid
# pyrefly: ignore [missing-import]
import SimpleITK as sitk
# pyrefly: ignore [missing-import]
from scipy.ndimage import distance_transform_edt
# pyrefly: ignore [missing-import]
from skimage.draw import polygon
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ConcatItemsd, ToTensord, SpatialPadd, DeleteItemsd
)
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader

# Must match training configuration
TARGET_SPACING = (1.27, 1.27, 2.5)  # mm
PATCH_SIZE = (96, 96, 96)

# Minimum padding for sliding window (must be divisible by patch size for efficiency)
MIN_PADDING = (16, 16, 16)
MODEL_PATH = "best_dose_model.pth"
CHANNELS = ["0000", "0001", "0002", "0003"]

# Structure matching patterns (same as dicom_to_nnunet.py)
# Using exact ROI names only
STRUCTURE_PATTERNS = {
    "PTV": [
        r"^CTV_62/20$",         # Only exact CTV_62/20
    ],
    "Bladder": [
        r"^Bladder$",           # Only exact Bladder
    ],
    "Anorectum": [
        r"^Anorectum$",         # Only exact Anorectum
    ],
}


def find_dicom_subdir(patient_dir):
    """Find the study subdirectory containing DICOM files."""
    subdirs = [d for d in os.listdir(patient_dir)
               if os.path.isdir(os.path.join(patient_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(patient_dir, subdirs[0])
    return patient_dir


def find_rtstruct(dicom_dir):
    """Find RTSTRUCT file in directory."""
    struct_files = glob.glob(os.path.join(dicom_dir, "*RTSTRUCT*"))
    if not struct_files:
        struct_files = glob.glob(os.path.join(dicom_dir, "*struct*"))
    if not struct_files:
        raise FileNotFoundError("No RTSTRUCT file found in directory")
    return struct_files[0]


def match_structure_name(roi_names, structure_type):
    """Match ROI name using regex patterns. Case-sensitive exact match."""
    patterns = STRUCTURE_PATTERNS[structure_type]
    for pattern in patterns:
        for name in roi_names:
            if re.match(pattern, name):  # Case-sensitive exact match
                return name
    return None


def load_ct(dicom_dir):
    """Load CT volume from DICOM series."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        raise FileNotFoundError("No DICOM series found")
    
    # Select series with most files (CT, not RTSTRUCT/RTDOSE)
    best_series = None
    best_count = 0
    for sid in series_ids:
        fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(fnames) > best_count:
            best_count = len(fnames)
            best_series = fnames
    
    reader.SetFileNames(best_series)
    ct_image = reader.Execute()
    
    if ct_image.GetDimension() != 3:
        raise ValueError(f"CT is {ct_image.GetDimension()}D, expected 3D")
    
    return ct_image


def rtstruct_to_mask(rs_ds, roi_name, ct_image):
    """Convert RTStruct contour to binary mask."""
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask = np.zeros(ct_array.shape, dtype=np.uint8)
    
    # Find ROI number
    roi_number = None
    for roi in rs_ds.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber
            break
    
    if roi_number is None:
        print(f"    WARNING: ROI '{roi_name}' not found")
        return mask
    
    # Find contour sequence
    contour_seq = None
    for roi_contour in rs_ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            contour_seq = roi_contour
            break
    
    if contour_seq is None or not hasattr(contour_seq, 'ContourSequence'):
        print(f"    WARNING: No contour data for '{roi_name}'")
        return mask
    
    # Rasterize contours
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


def compute_sdm(binary_mask, spacing_mm):
    """Compute signed distance map."""
    if binary_mask.sum() == 0:
        return np.ones_like(binary_mask, dtype=np.float32) * 100.0
    
    dist_outside = distance_transform_edt(binary_mask == 0, sampling=spacing_mm)
    dist_inside = distance_transform_edt(binary_mask == 1, sampling=spacing_mm)
    signed_dist = dist_outside - dist_inside
    
    return signed_dist.astype(np.float32)


def preprocess_dicom(dicom_dir):
    """
    Preprocess DICOM CT + RTSTRUCT into 4-channel input.
    
    Returns:
        ct_image: SimpleITK CT image (for spatial metadata)
        inputs_tensor: torch tensor [1, 4, D, H, W] ready for model
        original_shape: tuple of original CT shape before padding (D, H, W)
    """
    print("[1/4] Loading CT volume...")
    ct_image = load_ct(dicom_dir)
    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing = ct_image.GetSpacing()
    print(f"      CT shape: {ct_array.shape}, spacing: {spacing}")
    
    print("[2/4] Loading RTStruct...")
    rs_file = find_rtstruct(dicom_dir)
    rs_ds = pydicom.dcmread(rs_file)
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]
    print(f"      Found {len(roi_names)} ROIs")
    
    # Match structures
    ptv_name = match_structure_name(roi_names, "PTV")
    bladder_name = match_structure_name(roi_names, "Bladder")
    anorectum_name = match_structure_name(roi_names, "Anorectum")
    
    print(f"      PTV: {ptv_name}, Bladder: {bladder_name}, Anorectum: {anorectum_name}")
    
    if not all([ptv_name, bladder_name, anorectum_name]):
        raise ValueError(f"Could not match all structures. Available: {roi_names}")
    
    print("[3/4] Creating masks and SDMs...")
    ptv_mask = rtstruct_to_mask(rs_ds, ptv_name, ct_image)
    bladder_mask = rtstruct_to_mask(rs_ds, bladder_name, ct_image)
    anorectum_mask = rtstruct_to_mask(rs_ds, anorectum_name, ct_image)
    
    print(f"      PTV voxels: {ptv_mask.sum():,}")
    print(f"      Bladder voxels: {bladder_mask.sum():,}")
    print(f"      Anorectum voxels: {anorectum_mask.sum():,}")
    
    if ptv_mask.sum() == 0:
        raise ValueError("PTV mask is empty")
    
    # Compute SDMs
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    bladder_sdm = compute_sdm(bladder_mask, spacing_zyx)
    anorectum_sdm = compute_sdm(anorectum_mask, spacing_zyx)
    
    print(f"      Bladder SDM range: [{bladder_sdm.min():.1f}, {bladder_sdm.max():.1f}] mm")
    print(f"      Anorectum SDM range: [{anorectum_sdm.min():.1f}, {anorectum_sdm.max():.1f}] mm")
    
    # Save temporary NIfTI files for MONAI transforms
    temp_dir = os.path.join(dicom_dir, ".temp_inference")
    os.makedirs(temp_dir, exist_ok=True)
    
    def numpy_to_sitk(array, ref):
        img = sitk.GetImageFromArray(array)
        img.SetOrigin(ref.GetOrigin())
        img.SetSpacing(ref.GetSpacing())
        img.SetDirection(ref.GetDirection())
        return img
    
    # Write temporary files
    sitk.WriteImage(numpy_to_sitk(ct_array.astype(np.float32), ct_image),
                    os.path.join(temp_dir, "ch_0.nii.gz"))
    sitk.WriteImage(numpy_to_sitk(ptv_mask.astype(np.float32), ct_image),
                    os.path.join(temp_dir, "ch_1.nii.gz"))
    sitk.WriteImage(numpy_to_sitk(bladder_sdm, ct_image),
                    os.path.join(temp_dir, "ch_2.nii.gz"))
    sitk.WriteImage(numpy_to_sitk(anorectum_sdm, ct_image),
                    os.path.join(temp_dir, "ch_3.nii.gz"))
    
    # Calculate padded size that covers original + padding for sliding window
    # Need to ensure dimensions work with sliding window inference
    current_shape = ct_array.shape  # (D, H, W)
    padded_shape = list(current_shape)
    for i in range(3):
        # Ensure at least PATCH_SIZE + some padding for sliding window overlap
        min_dim = PATCH_SIZE[i] + MIN_PADDING[i]
        if padded_shape[i] < min_dim:
            padded_shape[i] = min_dim
    
    print(f"[4/4] Applying MONAI transforms...")
    print(f"      Original shape: {current_shape}")
    print(f"      Padded shape for inference: {padded_shape}")
    
    transform = Compose([
        LoadImaged(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
        EnsureChannelFirstd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
        Spacingd(
            keys=["ch_0", "ch_1", "ch_2", "ch_3"],
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear")
        ),
        # Use padded shape just large enough for sliding window
        SpatialPadd(
            keys=["ch_0", "ch_1", "ch_2", "ch_3"],
            spatial_size=padded_shape
        ),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
        DeleteItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
        ToTensord(keys=["image"])
    ])
    
    data_dict = {
        "ch_0": os.path.join(temp_dir, "ch_0.nii.gz"),
        "ch_1": os.path.join(temp_dir, "ch_1.nii.gz"),
        "ch_2": os.path.join(temp_dir, "ch_2.nii.gz"),
        "ch_3": os.path.join(temp_dir, "ch_3.nii.gz"),
    }
    
    ds = Dataset(data=[data_dict], transform=transform)
    loader = DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    inputs = batch["image"]  # [1, 4, D, H, W]
    
    print(f"      Input tensor shape: {inputs.shape}")
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    return ct_image, inputs, tuple(current_shape)


def create_rtdose_dicom(predicted_dose, ct_image, reference_dicom_dir, output_path, patient_name=""):
    """
    Create RTDOSE DICOM file from predicted dose array.
    
    Args:
        predicted_dose: numpy array [D, H, W] in Gy
        ct_image: SimpleITK CT image (for spatial alignment)
        reference_dicom_dir: directory with reference DICOMs (for metadata)
        output_path: path to save RTDOSE file
        patient_name: patient name for DICOM header
    """
    print("\n[Creating RTDOSE DICOM...]")
    
    # Get spatial info from CT
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()
    direction = ct_image.GetDirection()
    
    # Create new DICOM dataset using FileDataset
    # Generate UIDs
    sop_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    frame_of_ref_uid = generate_uid()
    
    # Create file meta information
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    
    # Create the main dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b'\x00' * 128)
    
    # Patient Information (copy from reference if available)
    ct_files = glob.glob(os.path.join(reference_dicom_dir, "*CT*")) or \
               glob.glob(os.path.join(reference_dicom_dir, "*.dcm"))
    if ct_files:
        ref_ds = pydicom.dcmread(ct_files[0], stop_before_pixels=True)
        ds.PatientName = getattr(ref_ds, 'PatientName', patient_name or "Anonymous")
        ds.PatientID = getattr(ref_ds, 'PatientID', "Unknown")
        ds.PatientBirthDate = getattr(ref_ds, 'PatientBirthDate', "")
        ds.PatientSex = getattr(ref_ds, 'PatientSex', "")
    else:
        ds.PatientName = patient_name or "Anonymous"
        ds.PatientID = "Unknown"
    
    # Study Information
    if ct_files:
        study_instance_uid = getattr(ref_ds, 'StudyInstanceUID', study_instance_uid)
        frame_of_ref_uid = getattr(ref_ds, 'FrameOfReferenceUID', frame_of_ref_uid)
    
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "AI Predicted Dose"
    
    # Series Information
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = "1"
    ds.Modality = "RTDOSE"
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = sop_instance_uid
    
    # Dose Grid Information
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    
    # Grid parameters (DICOM uses [row, column] = [Y, X] order)
    ds.Rows = predicted_dose.shape[1]  # Height
    ds.Columns = predicted_dose.shape[2]  # Width
    ds.NumberOfFrames = predicted_dose.shape[0]  # Depth (slices)
    
    # Pixel spacing in [Y, X] order
    ds.PixelSpacing = [spacing[1], spacing[0]]
    ds.GridFrameOffsetVector = [i * spacing[2] for i in range(predicted_dose.shape[0])]
    
    # Image position (origin in [X, Y, Z])
    ds.ImagePositionPatient = [origin[0], origin[1], origin[2]]
    ds.ImageOrientationPatient = [
        direction[0], direction[1], direction[2],
        direction[3], direction[4], direction[5]
    ]
    
    ds.FrameOfReferenceUID = frame_of_ref_uid
    
    # Pixel Data - convert to required format
    # DICOM dose is typically stored as scaled integers
    dose_max = predicted_dose.max()
    if dose_max > 0:
        # Scale to fit in uint16, keeping some precision
        dose_scaling = dose_max / 65535.0
        dose_scaled = (predicted_dose / dose_scaling).astype(np.uint16)
    else:
        dose_scaling = 1.0
        dose_scaled = predicted_dose.astype(np.uint16)
    
    ds.DoseGridScaling = str(dose_scaling)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    
    # Set pixel data (DICOM expects bytes)
    ds.PixelData = dose_scaled.tobytes()
    
    # Save
    ds.save_as(output_path)
    print(f"      Saved RTDOSE: {output_path}")
    print(f"      Dose grid: {predicted_dose.shape}")
    print(f"      Dose range: [{predicted_dose.min():.2f}, {predicted_dose.max():.2f}] Gy")
    print(f"      Dose scaling: {dose_scaling:.6f}")
    
    return output_path


def run_dicom_inference(dicom_dir, output_dose_path, model_path=None):
    """
    End-to-end inference: DICOM -> Predicted RTDOSE.
    
    Args:
        dicom_dir: directory containing CT DICOM series and RTSTRUCT
        output_dose_path: path for output RTDOSE DICOM file
        model_path: path to model checkpoint (default: best_dose_model.pth)
    
    Returns:
        output_path: path to saved RTDOSE file
        metadata: dict with dose statistics
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    model_path = model_path or MODEL_PATH
    print(f"Loading model from {model_path}...")
    
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!\n")
    
    # Preprocess DICOM
    ct_image, inputs, original_shape = preprocess_dicom(dicom_dir)
    inputs = inputs.to(device)
    
    # Run inference
    print("\n[Running inference...]")
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=PATCH_SIZE,
                sw_batch_size=4,
                predictor=model,
                overlap=0.25
            )
    
    # Extract predicted dose
    predicted_dose = outputs[0, 0].cpu().numpy() * 60.0  # [D, H, W]
    
    # Crop back to original shape after inference
    # predicted_dose from model is [D, H, W] after MONAI transforms
    # Need to crop to match original CT dimensions
    print(f"Prediction complete (padded shape)!")
    print(f"  Shape: {predicted_dose.shape}")
    print(f"  Range: [{predicted_dose.min():.2f}, {predicted_dose.max():.2f}] Gy")
    
    # Crop to original CT shape (after spacing transformation)
    # The Spacingd transform may change dimensions, so we crop to the post-spacing size
    crop_d = min(predicted_dose.shape[0], original_shape[0])
    crop_h = min(predicted_dose.shape[1], original_shape[1])
    crop_w = min(predicted_dose.shape[2], original_shape[2])
    
    predicted_dose_cropped = predicted_dose[:crop_d, :crop_h, :crop_w]
    
    print(f"  Cropped to: {predicted_dose_cropped.shape}")
    print(f"  Cropped range: [{predicted_dose_cropped.min():.2f}, {predicted_dose_cropped.max():.2f}] Gy")
    
    # Create RTDOSE DICOM with cropped dose
    output_path = create_rtdose_dicom(
        predicted_dose_cropped, 
        ct_image, 
        dicom_dir, 
        output_dose_path,
        patient_name=os.path.basename(dicom_dir)
    )
    
    metadata = {
        "dose_shape": predicted_dose.shape,
        "dose_min": float(predicted_dose.min()),
        "dose_max": float(predicted_dose.max()),
        "dose_mean": float(predicted_dose.mean()),
        "output_path": output_path,
    }
    
    return output_path, metadata


def main():
    parser = argparse.ArgumentParser(
        description="DICOM to RTDOSE inference for dose prediction"
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing CT DICOM series and RTSTRUCT"
    )
    parser.add_argument(
        "--output-dose", "-o",
        required=True,
        help="Output path for RTDOSE DICOM file (e.g., predicted_dose.dcm)"
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help=f"Path to model checkpoint (default: {MODEL_PATH})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DICOM Dose Prediction Inference")
    print("=" * 60)
    
    try:
        output_path, metadata = run_dicom_inference(
            args.input_dir,
            args.output_dose,
            args.model
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Output RTDOSE: {output_path}")
        print(f"Dose statistics:")
        print(f"  Min:  {metadata['dose_min']:.2f} Gy")
        print(f"  Max:  {metadata['dose_max']:.2f} Gy")
        print(f"  Mean: {metadata['dose_mean']:.2f} Gy")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
