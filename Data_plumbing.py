# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    NormalizeIntensityd, ConcatItemsd, ToTensord
)
# pyrefly: ignore [missing-import]
from monai.data import DataLoader, Dataset

# 1. Minimal Config
DATA_DIR = "/home/ankit/Dose_pred/nnUNet_raw/Dataset001_ProstateDose"
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")
TARGET_SPACING = (1.27, 1.27, 2.5)

# 2. Lean Transform Pipeline
test_transforms = Compose([
    LoadImaged(keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"]),
    EnsureChannelFirstd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"]),
    Spacingd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"],
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear")
    ),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    ToTensord(keys=["image", "dose_label"])
])

# 3. Target just Patient 000
patient_id = "prostate_000"
pt_dict = {
    "dose_label": os.path.join(LABELS_DIR, f"{patient_id}.nii.gz"),
    "ch_0": os.path.join(IMAGES_DIR, f"{patient_id}_0000.nii.gz"),
    "ch_1": os.path.join(IMAGES_DIR, f"{patient_id}_0001.nii.gz"),
    "ch_2": os.path.join(IMAGES_DIR, f"{patient_id}_0002.nii.gz"),
    "ch_3": os.path.join(IMAGES_DIR, f"{patient_id}_0003.nii.gz"),
}

print(f"Attempting to load and transform {patient_id}...")

try:
    # Use standard Dataset (NOT CacheDataset) for CPU
    ds = Dataset(data=[pt_dict], transform=test_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    
    batch = next(iter(loader))
    inputs = batch["image"]
    targets = batch["dose_label"]
    
    print("\n--- DATA PLUMBING SUCCESS! ---")
    print(f"Input Tensor Shape (Should be [1, 4, D, H, W]): {inputs.shape}")
    print(f"Target Tensor Shape (Should be [1, 1, D, H, W]): {targets.shape}")
    print(f"Input Value Range: {inputs.min():.2f} to {inputs.max():.2f}")

except Exception as e:
    print(f"\n--- DATA ERROR ---")
    print(f"Check your file paths or RAM: {e}")