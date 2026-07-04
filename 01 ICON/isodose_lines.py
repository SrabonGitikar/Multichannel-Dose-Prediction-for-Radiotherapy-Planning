import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def load_dose(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    dose_array = dcm.pixel_array * dcm.DoseGridScaling
    
    # Extract spatial metadata for registration
    origin = np.array(dcm.ImagePositionPatient, dtype=float)
    spacing = np.array([dcm.SliceThickness] + list(dcm.PixelSpacing), dtype=float) 
    
    # Construct physical coordinate grids
    z = origin[2] + np.arange(dose_array.shape[0]) * spacing[0]
    y = origin[1] + np.arange(dose_array.shape[1]) * spacing[1]
    x = origin[0] + np.arange(dose_array.shape[2]) * spacing[2]
    
    return dose_array, (z, y, x)

# 1. Load the Data
eclipse_dose, eclipse_coords = load_dose("path/to/eclipse_rtdose.dcm")
model_dose, model_coords = load_dose("path/to/model_rtdose.dcm")

# Note: You must load the corresponding CT volume and construct its coordinates similarly.
# For simplicity, assuming 'ct_array' and 'ct_coords' are loaded here.

# 2. Select the Isocenter Z-Slice (Axial)
# Find the index in the CT grid that corresponds to the PTV center
slice_idx = 50 # Example z-index

# 3. Setup the Multi-Panel Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
prescription = 36.25 # Gy for SBRT Experimental Arm
levels = [prescription * 0.5, prescription * 0.95, prescription * 1.0]
colors = ['blue', 'green', 'red']

# Plot Eclipse (Left)
axes[0].imshow(ct_array[slice_idx, :, :], cmap='gray', interpolation='none')
axes[0].contour(eclipse_dose[slice_idx, :, :], levels=levels, colors=colors, linewidths=1.5)
axes[0].set_title("Eclipse TPS (Ground Truth)")
axes[0].axis('off')

# Plot Dose-PlanNet (Right)
axes[1].imshow(ct_array[slice_idx, :, :], cmap='gray', interpolation='none')
axes[1].contour(model_dose[slice_idx, :, :], levels=levels, colors=colors, linewidths=1.5)
axes[1].set_title("Dose-PlanNet")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("spatial_dose_distribution.png", format='png', bbox_inches='tight')
plt.show()
