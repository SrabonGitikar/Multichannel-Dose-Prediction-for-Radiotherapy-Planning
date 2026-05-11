import os
import glob
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import monai
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    ToTensord,
    ConcatItemsd
)
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim

# 1. Configuration 
# Defaulting to relative path so it doesn't break on AWS. Override via env var.
DATA_DIR = os.environ.get("DATA_DIR", "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

# We want 4 input channels:
# 0000 = CT
# 0001 = PTV
# 0002 = Bladder SDM
# 0003 = Anorectum SDM
CHANNELS = ["0000", "0001", "0002", "0003"]
TARGET_SPACING = (1.27, 1.27, 2.5)  # Physical mm
PATCH_SIZE = (96, 96, 96)           # 3D Patch size for training


# 2. Dataset Setup
def get_data_dicts():
    # Find all patients by looking at the label files
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.nii.gz")))
    data_dicts = []
    
    for label_path in label_files:
        patient_id = os.path.basename(label_path).replace(".nii.gz", "")
        
        # Build dictionary of inputs for this patient
        pt_dict = {
            "dose_label": label_path,
        }
        
        # Add all 4 channels
        for i, ch in enumerate(CHANNELS):
            pt_dict[f"ch_{i}"] = os.path.join(IMAGES_DIR, f"{patient_id}_{ch}.nii.gz")
            
        data_dicts.append(pt_dict)
        
    return data_dicts

# Transforms Pipeline
train_transforms = Compose([
    # Load all NIfTI files
    LoadImaged(keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"]),
    
    # Ensure (C, H, W, D) format
    EnsureChannelFirstd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"]),
    
    # Resample everything to uniform spacing
    Spacingd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"],
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear")
    ),
    
    # Normalize CT Hounsfield Units only (ch_0)
    # We leave masks and SDMs as they are physically meaningful
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    
    # Concatenate the 4 input channels into a single "image" tensor
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    
    # Randomly crop 3D patches from the patient volume
    RandSpatialCropd(
        keys=["image", "dose_label"],
        roi_size=PATCH_SIZE,
        random_center=True,
        random_size=False
    ),
    
    # Convert to PyTorch Tensors
    ToTensord(keys=["image", "dose_label"])
])

# Validation Transforms (No random cropping for full-volume evaluation)
val_transforms = Compose([
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

# 3. Clinical Loss Function & Physics Hooks
class ClinicalDoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Baseline MSE
        self.mse = nn.MSELoss()
        
    def forward(self, pred_dose, true_dose):
        # We can expand this later with DVH penalties or organ-specific weights
        loss = self.mse(pred_dose, true_dose)
        return loss

# 4. Main Training Setup
def main():
    print("Finding data...")
    data_dicts = get_data_dicts()
    print(f"Found {len(data_dicts)} patients.")
    
    # Simple split (9 train, 2 val)
    train_files = data_dicts[:9]
    val_files = data_dicts[9:]
    
    print("Building datasets (caching to disk to save memory)...")
    cache_dir = os.path.join(DATA_DIR, "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
    # Reduced batch size and workers for 12GB GPU limits
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
    # Construct Custom MONAI U-Net
    print("Building 3D U-Net...")
    # Automatically select GPU if available (essential for H100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(
        spatial_dims=3,
        in_channels=4,        # 4 inputs
        out_channels=1,       # 1 continuous dose output
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # Optimizer and Loss
    loss_function = ClinicalDoseLoss()
    optimizer = optim.Adam(model.parameters(), 1e-4)
    
    # Mixed precision scaler for H100 speedup
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    print(f"Model and dataloaders ready on {device}!")
    
    # Simple Training Loop (Dummy run for local testing)
    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        step = 0
        
        for batch in train_loader:
            step += 1
            inputs = batch["image"].to(device)
            targets = batch["dose_label"].to(device)
            
            optimizer.zero_grad()
            
            # Autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            print(f"  Step {step}/{len(train_loader)} - Train Loss: {loss.item():.4f}")
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_d95_sum = 0.0
        val_bladder_mean_sum = 0.0
        val_rectum_mean_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                targets = batch["dose_label"].to(device)
                
                # Full volume inference using sliding window
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    outputs = sliding_window_inference(
                        inputs=inputs, 
                        roi_size=PATCH_SIZE, 
                        sw_batch_size=1,  # Reduced to 1 for 12GB GPU
                        predictor=model,
                        overlap=0.25
                    )
                
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                
                # Calculate Clinical Metrics
                ptv_mask = inputs[:, 1:2, ...] == 1.0
                bladder_mask = inputs[:, 2:3, ...] <= 0.0
                rectum_mask = inputs[:, 3:4, ...] <= 0.0
                
                # PTV D95 (5th percentile of dose inside PTV)
                ptv_dose = outputs[ptv_mask]
                if len(ptv_dose) > 0:
                    d95 = torch.quantile(ptv_dose, 0.05).item()
                else:
                    d95 = 0.0
                val_d95_sum += d95
                
                # Mean OAR Dose
                bladder_dose = outputs[bladder_mask]
                val_bladder_mean_sum += bladder_dose.mean().item() if len(bladder_dose) > 0 else 0.0
                
                rectum_dose = outputs[rectum_mask]
                val_rectum_mean_sum += rectum_dose.mean().item() if len(rectum_dose) > 0 else 0.0
                
        val_loss /= len(val_loader)
        avg_d95 = val_d95_sum / len(val_loader)
        avg_bladder = val_bladder_mean_sum / len(val_loader)
        avg_rectum = val_rectum_mean_sum / len(val_loader)
        
        print(f"  --> Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        print(f"      Validation Clinical Metrics: PTV D95: {avg_d95:.2f} Gy | Bladder Mean: {avg_bladder:.2f} Gy | Rectum Mean: {avg_rectum:.2f} Gy")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_dose_model.pth")
            print("  --> Saved new best model!")

if __name__ == "__main__":
    main()
