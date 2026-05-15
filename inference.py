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
    SpatialPadd,
    DeleteItemsd
)
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader

# Configuration - uses same paths as training
DATA_DIR = os.path.join(os.getcwd(), "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
TARGET_SPACING = (1.27, 1.27, 2.5)  # Physical mm - must match training
PATCH_SIZE = (96, 96, 96)  # Must match training
MODEL_PATH = "best_dose_model.pth"

# Input channels (must match training)
CHANNELS = ["0000", "0001", "0002", "0003"]  # CT, PTV, Bladder SDM, Anorectum SDM

# Inference Transforms (Same as training validation, but no label)
# Input: 4 NIfTI files (CT, PTV mask, Bladder SDM, Anorectum SDM)
# Output: 4-channel tensor for model input
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

def run_inference(patient_id, output_dir=".", save_nifti=True):
    """
    Run dose prediction inference on a single patient.
    
    Args:
        patient_id: Patient ID (e.g., "prostate_000")
        output_dir: Directory to save output files
        save_nifti: Whether to save output as NIfTI file
    
    Returns:
        pred_dose: numpy array of predicted dose [D, H, W] in Gy
        metadata: dict with spacing and patient info
    """
    # Auto-select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading model on {device}...")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first.")
    
    model.eval()

    print(f"\nRunning inference for {patient_id}...")
    
    # Build input dictionary with 4 channels
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
    
    print(f"Input tensor shape: {inputs.shape}")  # [1, 4, D, H, W]
    
    # Run inference with mixed precision if on GPU
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = sliding_window_inference(
                inputs=inputs, 
                roi_size=PATCH_SIZE, 
                sw_batch_size=4, 
                predictor=model,
                overlap=0.25
            )
    
    # outputs shape: [1, 1, H, W, D] (MONAI spatial order after Spacingd)
    # Transpose to (D, H, W) = (Z, Y, X) to match SimpleITK/NIfTI convention
    pred_dose = outputs[0, 0].cpu().numpy()          # (H, W, D)
    pred_dose = np.transpose(pred_dose, (2, 0, 1))   # -> (D, H, W) = (Z, Y, X)
    pred_dose = pred_dose * 60.0                     # denormalise to Gy
    pred_dose = np.clip(pred_dose, 0.0, None)        # dose cannot be negative
    
    print(f"Prediction complete. Shape: {pred_dose.shape}")
    print(f"Dose range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")
    
    # Save to NIfTI with correct spatial metadata from resampled input
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{patient_id}_predicted_dose.nii.gz")
        
        sitk_img = sitk.GetImageFromArray(pred_dose.astype(np.float32))
        sitk_img.SetSpacing(TARGET_SPACING)
        sitk.WriteImage(sitk_img, out_file)
        
        print(f"Saved predicted dose to: {out_file}")
    
    metadata = {
        "patient_id": patient_id,
        "spacing": TARGET_SPACING,
        "shape": pred_dose.shape,
        "dose_min": float(pred_dose.min()),
        "dose_max": float(pred_dose.max()),
    }
    
    return pred_dose, metadata


def main():
    global MODEL_PATH
    
    parser = argparse.ArgumentParser(description="Dose prediction inference")
    parser.add_argument("--patient", type=str, required=True, 
                        help="Patient ID (e.g., prostate_000)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for predicted dose")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    MODEL_PATH = args.model
    
    try:
        pred_dose, metadata = run_inference(args.patient, args.output_dir)
        print("\n=== Inference Summary ===")
        print(f"Patient: {metadata['patient_id']}")
        print(f"Output shape: {metadata['shape']}")
        print(f"Dose range: [{metadata['dose_min']:.2f}, {metadata['dose_max']:.2f}] Gy")
        print("=========================")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
