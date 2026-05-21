import os
import glob
import argparse
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
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
PATCH_SIZE = (128, 128, 64)  # Must match training
MODEL_PATH = "best_dose_model_physics.pth"
PRESCRIPTION_DOSE_GY = 75.0

# Input channels — must stay in sync with CHANNELS in train_dummy_physics_new.py
CHANNELS = ["0000", "0001", "0002", "0003", "0004", "0005"]  # CT, PTV, Bladder SDM, Anorectum SDM, Beam, Body Mask

# Spacingd interpolation modes — binary masks use 'nearest', continuous fields 'bilinear'
# Order matches CHANNELS: CT=bilinear, PTV=nearest, Bladder SDM=bilinear,
#   Anorectum SDM=bilinear, Beam Mask=nearest, Body Mask=nearest
_CH_KEYS  = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5"]
_CH_MODES = ("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest")

inference_transforms = Compose([
    LoadImaged(keys=_CH_KEYS),
    EnsureChannelFirstd(keys=_CH_KEYS),
    Spacingd(
        keys=_CH_KEYS,
        pixdim=TARGET_SPACING,
        mode=_CH_MODES,
    ),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=_CH_KEYS, name="image"),
    DeleteItemsd(keys=_CH_KEYS),
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
        in_channels=6,  # CT, PTV, Bladder SDM, Anorectum SDM, Beam Prior, Body Mask
        out_channels=1,
        channels=(16, 32, 64, 128),  # 4-level UNet — must match training
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first.")
    
    model.eval()

    print(f"\nRunning inference for {patient_id}...")
    
    # Build input dictionary with all 6 channels
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

    # Capture spatial metadata from the resampled CT (channel 0) for NIfTI alignment
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
    grid_origin    = ct_resampled.GetOrigin()
    grid_direction = ct_resampled.GetDirection()

    print(f"Input tensor shape: {inputs.shape}")  # [1, 6, D, H, W]

    # Run inference with sliding window + mixed precision
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=PATCH_SIZE,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )

    # Apply Softplus outside autocast in float32 — matches the training loop exactly.
    # (float16 saturates at ~65504; Softplus inside AMP could silently overflow.)
    # MONAI's Spacingd preserves (Z, Y, X) = (D, H, W) ordering, so no transpose needed.
    outputs_activated = F.softplus(outputs.float())  # (1, 1, D, H, W)

    # Hard-zero dose outside the patient body — mirrors training loop.
    # ch_5 = body mask: 1.0 inside body (CT > -300 HU), 0.0 in air.
    body_mask_hard = (inputs[:, 5:6, ...] > 0.5).float()
    outputs_activated = outputs_activated * body_mask_hard

    pred_dose = outputs_activated[0, 0].cpu().numpy()  # (D, H, W)
    pred_dose = pred_dose * PRESCRIPTION_DOSE_GY        # denormalise to Gy
    pred_dose = np.clip(pred_dose, 0.0, None)           # dose cannot be negative
    
    print(f"Prediction complete. Shape: {pred_dose.shape}")
    print(f"Dose range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")
    
    # Save to NIfTI with correct spatial metadata from resampled input
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{patient_id}_predicted_dose.nii.gz")
        
        sitk_img = sitk.GetImageFromArray(pred_dose.astype(np.float32))
        sitk_img.SetSpacing(TARGET_SPACING)
        sitk_img.SetOrigin(grid_origin)
        sitk_img.SetDirection(grid_direction)
        sitk.WriteImage(sitk_img, out_file)
        
        print(f"Saved predicted dose to: {out_file}")
    
    metadata = {
        "patient_id": patient_id,
        "spacing": TARGET_SPACING,
        "origin": grid_origin,
        "direction": grid_direction,
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
        print("\n=== Dose Grid Summary ===")
        print(f"Patient:   {metadata['patient_id']}")
        print(f"Shape:     {metadata['shape']}  (Z, Y, X voxels)")
        print(f"Spacing:   {metadata['spacing']} mm")
        print(f"Origin:    {tuple(round(v,2) for v in metadata['origin'])} mm")
        print(f"Dose range: [{metadata['dose_min']:.2f}, {metadata['dose_max']:.2f}] Gy")
        print("=========================")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
