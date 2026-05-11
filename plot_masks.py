import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import SimpleITK as sitk
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

DATA_DIR = "/home/ankit/Dose_pred/nnUNet_raw/Dataset001_ProstateDose"
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

patient_id = "prostate_000"

def load_nifti(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

print("Loading arrays...")
ct = load_nifti(os.path.join(IMAGES_DIR, f"{patient_id}_0000.nii.gz"))
ptv = load_nifti(os.path.join(IMAGES_DIR, f"{patient_id}_0001.nii.gz"))
bladder_sdm = load_nifti(os.path.join(IMAGES_DIR, f"{patient_id}_0002.nii.gz"))
rectum_sdm = load_nifti(os.path.join(IMAGES_DIR, f"{patient_id}_0003.nii.gz"))
dose = load_nifti(os.path.join(LABELS_DIR, f"{patient_id}.nii.gz"))

# Find a slice with PTV presence
z_slices = np.where(ptv == 1)[0]
if len(z_slices) > 0:
    mid_z = z_slices[len(z_slices) // 2]
else:
    mid_z = ct.shape[0] // 2

print(f"Generating plot for Z slice {mid_z}...")

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# CT
axes[0].imshow(ct[mid_z, :, :], cmap='gray', vmin=-200, vmax=300)
axes[0].set_title(f'CT (Slice {mid_z})')
axes[0].axis('off')

# PTV
axes[1].imshow(ct[mid_z, :, :], cmap='gray', vmin=-200, vmax=300)
axes[1].imshow(ptv[mid_z, :, :], cmap='Reds', alpha=0.5 * (ptv[mid_z, :, :]>0))
axes[1].set_title('PTV Mask Overlay')
axes[1].axis('off')

# Bladder SDM
im = axes[2].imshow(bladder_sdm[mid_z, :, :], cmap='coolwarm')
axes[2].contour(bladder_sdm[mid_z, :, :], levels=[0], colors='green', linewidths=2)
axes[2].set_title('Bladder SDM (Green=Boundary)')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Rectum SDM
im2 = axes[3].imshow(rectum_sdm[mid_z, :, :], cmap='coolwarm')
axes[3].contour(rectum_sdm[mid_z, :, :], levels=[0], colors='blue', linewidths=2)
axes[3].set_title('Rectum SDM (Blue=Boundary)')
axes[3].axis('off')
fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

# Dose
im3 = axes[4].imshow(dose[mid_z, :, :], cmap='jet')
axes[4].set_title('Target Dose Map (Gy)')
axes[4].axis('off')
fig.colorbar(im3, ax=axes[4], fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = '/home/ankit/Dose_pred/mask_verification.png'
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
