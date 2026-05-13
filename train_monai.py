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
    ConcatItemsd,
    SpatialPadd,
    DeleteItemsd
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
DATA_DIR = os.path.join(os.getcwd(), "./nnUNet_raw/Dataset001_ProstateDose")
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
    
    # Pad to ensure consistent size before cropping (handles variable patient dimensions)
    SpatialPadd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"],
        spatial_size=PATCH_SIZE
    ),
    
    # Normalize CT Hounsfield Units only (ch_0)
    # We leave masks and SDMs as they are physically meaningful
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    
    # Concatenate the 4 input channels into a single "image" tensor
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    
    # Remove individual channel keys to prevent collation errors
    DeleteItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
    
    # Randomly crop 3D patches from the patient volume
    RandSpatialCropd(
        keys=["image", "dose_label"],
        roi_size=PATCH_SIZE,
        random_center=True,
        random_size=False
    ),
    
    # Data augmentation: random flip
    RandFlipd(keys=["image", "dose_label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "dose_label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "dose_label"], spatial_axis=[2], prob=0.5),
    
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
    # Pad validation images to ensure consistent size for batching
    SpatialPadd(
        keys=["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"],
        spatial_size=(512, 512, 256)
    ),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
    # Remove individual channel keys to prevent collation errors
    DeleteItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"]),
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
        self.lambda_ptv = 5.0
        self.lambda_oar = 2.0
        self.lambda_smooth = 0.5
        
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
            bladder_mean = torch.mean(bladder_pred)
            # Penalize only if mean dose exceeds max tolerance
            loss_oar += torch.relu(bladder_mean - self.max_bladder) ** 2
            
        if len(rectum_pred) > 0:
            rectum_mean = torch.mean(rectum_pred)
            # Penalize only if mean dose exceeds max tolerance
            loss_oar += torch.relu(rectum_mean - self.max_rectum) ** 2
            
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
    # Increased batch size and workers for AWS H100
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
    
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)
    # Batch size 1 for validation - images have different sizes after resampling
    # sliding_window_inference handles full volume prediction per patient
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
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
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    print(f"Model and dataloaders ready on {device}!")
    
    # Simple Training Loop
    epochs = 300
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
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(inputs)
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
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
                    outputs = sliding_window_inference(
                        inputs=inputs, 
                        roi_size=PATCH_SIZE, 
                        sw_batch_size=4, 
                        predictor=model,
                        overlap=0.25
                    )
                
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
