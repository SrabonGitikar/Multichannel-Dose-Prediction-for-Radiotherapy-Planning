import os
import glob
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
    # pyrefly: ignore [missing-import]
    Compose,
    # pyrefly: ignore [missing-import]
    LoadImaged,
    # pyrefly: ignore [missing-import]
    EnsureChannelFirstd,
    # pyrefly: ignore [missing-import]
    Spacingd,
    # pyrefly: ignore [missing-import]
    NormalizeIntensityd,
    # pyrefly: ignore [missing-import]
    ConcatItemsd,
    # pyrefly: ignore [missing-import]
    ToTensord
)
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader

# Configuration
DATA_DIR = "/home/ankit/Dose_pred/nnUNet_raw/Dataset001_ProstateDose"
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
TARGET_SPACING = (1.27, 1.27, 2.5)  # Physical mm
PATCH_SIZE = (96, 96, 96)
MODEL_PATH = "best_dose_model.pth"

# Validation Transforms (Same as train_monai.py, but no label)
val_transforms = Compose([
    LoadImaged(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    EnsureChannelFirstd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    Spacingd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3"],
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear")
    ),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    ToTensord(keys=["image"])
])

def main():
    # Force CPU because the Quadro K620 (compute capability 5.0) is not supported by modern PyTorch
    device = torch.device("cpu")
    
    # Load Model
    print("Loading model...")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        # Allow missing keys or shape mismatches just in case, but standard load
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully!")
    else:
        print(f"Warning: '{MODEL_PATH}' not found. Running with uninitialized random weights.")
    
    model.eval()

    # We will test on patient 000
    patient_id = "prostate_000"
    print(f"\nRunning inference for {patient_id}...")
    
    pt_dict = {}
    for i in range(4):
        pt_dict[f"ch_{i}"] = os.path.join(IMAGES_DIR, f"{patient_id}_000{i}.nii.gz")
        
    # Create dataset just for the transform pipeline
    ds = Dataset(data=[pt_dict], transform=val_transforms)
    loader = DataLoader(ds, batch_size=1)
    
    batch = next(iter(loader))
    inputs = batch["image"].to(device)
    
    with torch.no_grad():
        print("Applying sliding window inference...")
        outputs = sliding_window_inference(
            inputs=inputs, 
            roi_size=PATCH_SIZE, 
            sw_batch_size=4, 
            predictor=model,
            overlap=0.25
        )
        
    # Squeeze out batch and channel dimensions -> [D, H, W]
    pred_dose = outputs[0, 0].cpu().numpy()
    
    print(f"Prediction complete. Shape: {pred_dose.shape}, Range: [{pred_dose.min():.2f}, {pred_dose.max():.2f}] Gy")
    
    # Save to NIfTI
    out_file = f"{patient_id}_predicted_dose.nii.gz"
    
    # Note: We save the resampled grid directly. For clinical deployment, you would invert the Spacingd transform
    # to project it back onto the original CT coordinates. 
    sitk_img = sitk.GetImageFromArray(pred_dose)
    sitk_img.SetSpacing((TARGET_SPACING[0], TARGET_SPACING[1], TARGET_SPACING[2]))
    sitk.WriteImage(sitk_img, out_file)
    
    print(f"Saved predicted dose volume to {out_file}")

if __name__ == "__main__":
    main()
