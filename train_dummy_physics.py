import os
import glob
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import monai
# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset, list_data_collate
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
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
    
    # Targeted cropping: 50% tumor, 50% background to prevent global average local minimum
    RandCropByPosNegLabeld(
        keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"],
        label_key="ch_1", # PTV mask acts as the spatial anchor
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=4,
    ),
    
    # Concatenate the 4 input channels into a single "image" tensor
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    
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
    def __init__(self, d_prescription=60.0, max_bladder=40.0, max_rectum=45.0):
        super().__init__()
        # Baseline MSE
        self.mse = nn.MSELoss()
        
        # Clinical parameters (Set these to your exact clinical protocol)
        self.d_prescription = d_prescription
        self.max_bladder = max_bladder
        self.max_rectum = max_rectum
        
        # Hyperparameter weights (Lambdas from PDF)
        self.lambda_mse = 1.0
        self.lambda_ptv = 0.1
        self.lambda_oar = 0.05
        self.lambda_smooth = 0.01
        
    def forward(self, pred_dose, true_dose, inputs):
        # 1. L_MSE: Baseline Voxel-wise Reconstruction
        loss_mse = self.mse(pred_dose, true_dose)
        
        # 2. L_PTV: PTV Coverage Penalty
        # Extract PTV mask (Channel 1, 1.0 = inside PTV)
        ptv_mask = inputs[:, 1:2, ...] == 1.0
        ptv_pred = pred_dose[ptv_mask]
        
        if len(ptv_pred) > 0:
            # Penalize only if predicted dose is less than prescription (ReLU behavior)
            underdose_error = torch.relu(self.d_prescription - ptv_pred)
            loss_ptv = torch.mean(underdose_error ** 2)
        else:
            loss_ptv = torch.tensor(0.0, device=pred_dose.device)
            
        # 3. L_OAR: Organ-at-Risk Penalty
        # Extract Bladder (Channel 2) and Rectum (Channel 3) masks using SDM logic (<= 0 is inside organ)
        bladder_mask = inputs[:, 2:3, ...] <= 0.0
        rectum_mask = inputs[:, 3:4, ...] <= 0.0
        
        bladder_pred = pred_dose[bladder_mask]
        rectum_pred = pred_dose[rectum_mask]
        
        loss_oar = torch.tensor(0.0, device=pred_dose.device)
        
        if len(bladder_pred) > 0:
            # Voxel-wise penalty instead of patch-mean penalty
            overdose_voxels = torch.relu(bladder_pred - self.max_bladder)
            loss_oar += torch.mean(overdose_voxels ** 2)
            
        if len(rectum_pred) > 0:
            # Voxel-wise penalty instead of patch-mean penalty
            overdose_voxels = torch.relu(rectum_pred - self.max_rectum)
            loss_oar += torch.mean(overdose_voxels ** 2)
            
        # 4. L_smooth: Spatial Smoothness Constraint (Total Variation / Sobolev)
        # Calculate gradients along D, H, W axes
        grad_d = torch.abs(pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :])
        grad_h = torch.abs(pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :])
        grad_w = torch.abs(pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1])
        loss_smooth = torch.mean(grad_d**2) + torch.mean(grad_h**2) + torch.mean(grad_w**2)
        
        # Total Weighted PGNN Loss
        loss_total = (self.lambda_mse * loss_mse) + \
                     (self.lambda_ptv * loss_ptv) + \
                     (self.lambda_oar * loss_oar) + \
                     (self.lambda_smooth * loss_smooth)
                     
        return loss_total

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
    # Safest setting for 12GB GPU: 1 patient * 4 patches = 4 effective batch size
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=list_data_collate)
    
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
    
    # Simple Training Loop
    epochs = 100
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
                raw_outputs = model(inputs)
                outputs = torch.relu(raw_outputs)  # Strict Physics Constraint
                loss = loss_function(outputs, targets, inputs)
                
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
                        sw_batch_size=1, 
                        predictor=model,
                        overlap=0.25
                    )
                    outputs = torch.relu(outputs)  # Strict Physics Constraint
                
                loss = loss_function(outputs, targets, inputs)
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
