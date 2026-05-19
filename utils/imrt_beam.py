import os
# pyrefly: ignore [missing-import]
import pydicom
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import nibabel as nib

def generate_beam_prior(rtplan_path, ct_nifti_path, output_nifti_path, cylinder_radius_mm=50.0):
    print(f"Processing: {rtplan_path}")
    
    # 1. Load CT NIfTI to get spatial dimensions and Affine Matrix
    ct_img = nib.load(ct_nifti_path)
    affine = ct_img.affine
    inv_affine = np.linalg.inv(affine)
    shape = ct_img.shape
    
    # 2. Parse RTPlan DICOM
    plan = pydicom.dcmread(rtplan_path)
    
    isocenter_mm = None
    gantry_angles_deg = []
    
    # Extract beams that are actually used for treatment
    for beam in plan.BeamSequence:
        if beam.BeamType == "STATIC" or beam.TreatmentDeliveryType == "TREATMENT":
            cp0 = beam.ControlPointSequence[0]
            
            # Grab Isocenter from the first control point
            if isocenter_mm is None and hasattr(cp0, 'IsocenterPosition'):
                isocenter_mm = np.array(cp0.IsocenterPosition)
                
            if hasattr(cp0, 'GantryAngle'):
                gantry_angles_deg.append(float(cp0.GantryAngle))
                
    if isocenter_mm is None or not gantry_angles_deg:
        raise ValueError("Could not extract Isocenter or Gantry Angles from RTPlan.")
        
    print(f"  Found Isocenter (mm): {isocenter_mm}")
    print(f"  Found {len(gantry_angles_deg)} Gantry Angles: {gantry_angles_deg}")

    # 3. Convert Isocenter to Voxel Coordinates
    iso_homog = np.append(isocenter_mm, 1.0)
    iso_voxel = inv_affine.dot(iso_homog)[:3]
    
    # 4. Generate the 3D Voxel Grid
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert voxel grid to physical space (mm) for accurate distance calculations
    # Flatten the grid, multiply by affine, and reshape
    grid_coords = np.stack((X, Y, Z, np.ones_like(X)), axis=-1)
    physical_grid = np.einsum('ij,klmj->klmi', affine, grid_coords)[..., :3]
    
    # 5. Initialize the Binary Beam Mask
    beam_mask = np.zeros(shape, dtype=np.float32)
    
    # 6. Draw the Cylinders
    for angle in gantry_angles_deg:
        # Convert DICOM IEC 61217 angle to standard mathematical vector
        # Note: Depending on patient orientation (HFS/FFS), this vector might need axis flipping.
        # Standard HFS: Gantry 0 is facing anterior (y-axis).
        theta_rad = np.deg2rad(angle)
        
        # Direction vector of the beam
        beam_dir = np.array([np.sin(theta_rad), -np.cos(theta_rad), 0.0])
        beam_dir = beam_dir / np.linalg.norm(beam_dir)
        
        # Calculate perpendicular distance from every voxel to the beam line segment
        # Line eq: P = iso + t * dir. Distance = || (V - iso) - ((V - iso) dot dir) * dir ||
        vec_to_iso = physical_grid - isocenter_mm
        projection_length = np.sum(vec_to_iso * beam_dir, axis=-1)
        
        # Expand dims to subtract the projected vector
        projection_vec = projection_length[..., np.newaxis] * beam_dir
        perp_distance = np.linalg.norm(vec_to_iso - projection_vec, axis=-1)
        
        # Union the mask: 1 if within the cylinder radius, else keep existing value
        beam_mask[perp_distance <= cylinder_radius_mm] = 1.0

    # 7. Save as new NIfTI channel
    new_img = nib.Nifti1Image(beam_mask, affine)
    nib.save(new_img, output_nifti_path)
    print(f"  Saved 5th Channel to: {output_nifti_path}\n")

# Example Usage:
# generate_beam_prior(
#     rtplan_path="path/to/patient001/RTPlan.dcm", 
#     ct_nifti_path="imagesTr/patient001_0000.nii.gz", 
#     output_nifti_path="imagesTr/patient001_0004.nii.gz"
# )