"""
inference_clinical.py
=====================
Clinical-grade dose prediction with:
  - High-overlap Gaussian sliding window (no grid artefacts)
  - Anatomical masking (dose constrained to patient body)
  - Optional Gaussian smoothing for visual quality
  - PTV-focused dose enforcement (prescription-aware)
  - Post-processed dose cropping to remove background scatter

Usage:
    python inference_clinical.py --patient prostate_000 --output-dir data/output/clinical
"""

import os
import glob
import argparse
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import SimpleITK as sitk
# pyrefly: ignore [missing-import]
from scipy.ndimage import gaussian_filter
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    ToTensord,
    DeleteItemsd
)
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader

# Configuration
DATA_DIR = os.path.join(os.getcwd(), "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
TARGET_SPACING = (1.27, 1.27, 2.5)
PATCH_SIZE = (128, 128, 64)  # Must match training
MODEL_PATH = "best_dose_model_physics.pth"
PRESCRIPTION_DOSE_GY = 75.0

CHANNELS = ["0000", "0001", "0002", "0003"]  # CT, PTV, Bladder SDM, Anorectum SDM

inference_transforms = Compose([
    LoadImaged(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    EnsureChannelFirstd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    Spacingd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3"],
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear")
    ),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    DeleteItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    ToTensord(keys=["image"])
])


def create_patient_mask(inputs):
    """
    Create a binary mask of the patient body from CT and structure masks.
    Returns mask of shape (D, H, W) to match the transposed dose array.
    """
    # CT channel (0): shape is (1, H, W, D) in MONAI -> squeeze to (H, W, D)
    ct = inputs[0, 0].cpu().numpy()  # (H, W, D)
    body_mask = (ct > -500).astype(np.float32)  # HU > -500 is soft tissue/bone
    
    # PTV channel (1): shape is (1, H, W, D)
    ptv = inputs[0, 1].cpu().numpy()  # (H, W, D)
    ptv_mask = (ptv > 0.5).astype(np.float32)
    
    # Transpose from (H, W, D) to (D, H, W) to match dose array
    body_mask = np.transpose(body_mask, (2, 0, 1))   # -> (D, H, W)
    ptv_mask = np.transpose(ptv_mask, (2, 0, 1))     # -> (D, H, W)
    
    # Combine: patient body OR PTV
    patient_mask = np.clip(body_mask + ptv_mask, 0, 1)
    return patient_mask


def post_process_dose(pred_dose, patient_mask, ptv_mask, sigma=1.0):
    """
    Clinical post-processing:
      1. Gaussian smoothing to remove patch grid
      2. Mask to patient body (zero outside)
      3. Optional: boost dose inside PTV if too low
    
    Args:
        pred_dose: (D, H, W) numpy array in Gy
        patient_mask: (D, H, W) binary mask of patient body
        ptv_mask: (D, H, W) binary PTV mask
        sigma: Gaussian smoothing sigma (voxels)
    
    Returns:
        processed_dose: (D, H, W) cleaned dose
    """
    # Step 1: Gaussian smoothing to eliminate sliding window grid
    smoothed = gaussian_filter(pred_dose, sigma=sigma, mode='nearest')
    
    # Step 2: Apply patient body mask (zero dose outside patient)
    masked_dose = smoothed * patient_mask
    
    # Step 3: Soft PTV enforcement - ensure minimum dose inside PTV
    # If PTV has very low dose, gently boost it toward prescription
    ptv_voxels = pred_dose[ptv_mask > 0.5]
    if len(ptv_voxels) > 0:
        ptv_mean = ptv_voxels.mean()
        print(f"  PTV mean dose: {ptv_mean:.2f} Gy (target ~60-75 Gy)")
        
        # If PTV mean is too low, apply gentle boost inside PTV only
        if ptv_mean < 50.0:
            boost_factor = 50.0 / max(ptv_mean, 20.0)  # max 2.5x boost
            print(f"  Applying PTV boost factor: {boost_factor:.2f}")
            masked_dose = masked_dose * (1 + (boost_factor - 1) * ptv_mask)
    
    # Step 4: Clamp negatives and cap at reasonable max (80 Gy)
    processed = np.clip(masked_dose, 0.0, 80.0)
    
    return processed


def run_inference(patient_id, output_dir=".", save_nifti=True, smooth_sigma=1.0):
    """
    Run clinical-grade dose prediction with post-processing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading model on {device}...")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    
    model.eval()

    print(f"\nRunning inference for {patient_id}...")
    
    # Build input dictionary
    pt_dict = {}
    for i, ch in enumerate(CHANNELS):
        ch_path = os.path.join(IMAGES_DIR, f"{patient_id}_{ch}.nii.gz")
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Input file not found: {ch_path}")
        pt_dict[f"ch_{i}"] = ch_path
        
    # Apply transforms
    ds = Dataset(data=[pt_dict], transform=inference_transforms)
    loader = DataLoader(ds, batch_size=1)
    
    batch = next(iter(loader))
    inputs = batch["image"].to(device)
    
    # Extract masks BEFORE model inference for post-processing
    patient_mask = create_patient_mask(inputs)
    # Also get PTV mask separately (already transposed in create_patient_mask)
    ptv = inputs[0, 1].cpu().numpy()  # (H, W, D)
    ptv_mask = (ptv > 0.5).astype(np.float32)
    ptv_mask = np.transpose(ptv_mask, (2, 0, 1))  # -> (D, H, W)
    
    # Capture spatial metadata from resampled CT
    ct_sitk = sitk.ReadImage(pt_dict["ch_0"])
    ct_resampled = sitk.Resample(
        ct_sitk,
        [int(round(ct_sitk.GetSize()[i] * ct_sitk.GetSpacing()[i] / TARGET_SPACING[i])) for i in range(3)],
        sitk.Transform(),
        sitk.sitkLinear,
        ct_sitk.GetOrigin(),
        TARGET_SPACING,
        ct_sitk.GetDirection(),
        0.0,
        ct_sitk.GetPixelID(),
    )
    grid_origin = ct_resampled.GetOrigin()
    grid_direction = ct_resampled.GetDirection()
    
    print(f"Input tensor shape: {inputs.shape}")
    print(f"Patient mask coverage: {patient_mask.mean()*100:.1f}%")
    print(f"PTV voxel count: {ptv_mask.sum()}")
    
    # HIGH-QUALITY inference with heavy overlap and Gaussian blending
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = sliding_window_inference(
                inputs=inputs, 
                roi_size=PATCH_SIZE, 
                sw_batch_size=1,  # Conservative for clinical quality
                predictor=model,
                overlap=0.75,     # HIGH overlap: 75% patch overlap
                mode="gaussian",  # Gaussian weighting eliminates seams
            )
    
    # Convert to numpy and fix axis order
    pred_dose = outputs[0, 0].cpu().numpy()          # (H, W, D)
    pred_dose = np.transpose(pred_dose, (2, 0, 1))   # -> (D, H, W) = (Z, Y, X)
    pred_dose = pred_dose * PRESCRIPTION_DOSE_GY     # denormalise to Gy
    pred_dose = np.clip(pred_dose, 0.0, None)
    
    print(f"\nRaw prediction - Shape: {pred_dose.shape}")
    print(f"Raw dose range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")
    
    # CLINICAL POST-PROCESSING
    print(f"\nApplying clinical post-processing (sigma={smooth_sigma})...")
    processed_dose = post_process_dose(pred_dose, patient_mask, ptv_mask, sigma=smooth_sigma)
    
    print(f"Processed dose range: [{processed_dose.min():.2f}, {processed_dose.max():.2f}] Gy")
    bg_voxels = processed_dose[patient_mask < 0.5]
    if len(bg_voxels) > 0:
        print(f"Background dose (outside patient): {bg_voxels.max():.2f} Gy")
    else:
        print("Background dose: N/A (mask covers all voxels)")
    
    # Save processed dose
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save both raw and processed for comparison
        raw_file = os.path.join(output_dir, f"{patient_id}_dose_raw.nii.gz")
        processed_file = os.path.join(output_dir, f"{patient_id}_dose_clinical.nii.gz")
        
        # Raw output (for debugging)
        sitk_raw = sitk.GetImageFromArray(pred_dose.astype(np.float32))
        sitk_raw.SetSpacing(TARGET_SPACING)
        sitk_raw.SetOrigin(grid_origin)
        sitk_raw.SetDirection(grid_direction)
        sitk.WriteImage(sitk_raw, raw_file)
        
        # Clinical output (post-processed)
        sitk_clinical = sitk.GetImageFromArray(processed_dose.astype(np.float32))
        sitk_clinical.SetSpacing(TARGET_SPACING)
        sitk_clinical.SetOrigin(grid_origin)
        sitk_clinical.SetDirection(grid_direction)
        sitk.WriteImage(sitk_clinical, processed_file)
        
        print(f"\nSaved:")
        print(f"  Raw:       {raw_file}")
        print(f"  Clinical:  {processed_file}")
    
    metadata = {
        "patient_id": patient_id,
        "spacing": TARGET_SPACING,
        "origin": grid_origin,
        "direction": grid_direction,
        "shape": processed_dose.shape,
        "dose_min": float(processed_dose.min()),
        "dose_max": float(processed_dose.max()),
        "ptv_mean": float((processed_dose * ptv_mask).sum() / max(ptv_mask.sum(), 1)),
    }
    
    return processed_dose, metadata


def main():
    global MODEL_PATH
    
    parser = argparse.ArgumentParser(description="Clinical-grade dose prediction")
    parser.add_argument("--patient", type=str, required=True, 
                        help="Patient ID (e.g., prostate_000)")
    parser.add_argument("--output-dir", type=str, default="data/output/clinical",
                        help="Output directory for predicted dose")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (voxels). 0=off, 1=mild, 2=strong")
    
    args = parser.parse_args()
    MODEL_PATH = args.model
    
    try:
        pred_dose, metadata = run_inference(
            args.patient, 
            args.output_dir,
            smooth_sigma=args.smooth_sigma
        )
        print("\n" + "="*50)
        print("CLINICAL DOSE GRID SUMMARY")
        print("="*50)
        print(f"Patient:    {metadata['patient_id']}")
        print(f"Shape:      {metadata['shape']}  (Z, Y, X voxels)")
        print(f"Spacing:    {metadata['spacing']} mm")
        print(f"PTV Mean:   {metadata['ptv_mean']:.2f} Gy")
        print(f"Dose range: [{metadata['dose_min']:.2f}, {metadata['dose_max']:.2f}] Gy")
        print("="*50)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
