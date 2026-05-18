import os
import glob
from imrt_beam import generate_beam_prior

# ==============================================================================
# CONFIGURATION - Set these paths to your local environment
# ==============================================================================
# Directory containing your patient folders with raw DICOM files (including RTPlan)
RAW_DICOM_DIR = "/path/to/raw/dicom/dataset"

# Directory containing your preprocessed NIfTI images for nnU-Net
IMAGES_TR_DIR = "./nnUNet_raw/Dataset001_ProstateDose/imagesTr"

def main():
    print(f"Starting batch generation of IMRT beam priors...")
    
    # We assume each patient has a directory in RAW_DICOM_DIR named by patient ID
    # and their CT is already processed as {patient_id}_0000.nii.gz in IMAGES_TR_DIR
    ct_files = sorted(glob.glob(os.path.join(IMAGES_TR_DIR, "*_0000.nii.gz")))
    
    if not ct_files:
        print(f"No CT files found in {IMAGES_TR_DIR}")
        return

    success_count = 0
    error_count = 0

    for ct_path in ct_files:
        # Extract patient ID (e.g., 'prostate_001' from 'prostate_001_0000.nii.gz')
        basename = os.path.basename(ct_path)
        patient_id = basename.replace("_0000.nii.gz", "")
        
        # Locate the RTPlan DICOM file. Adjust this search pattern based on your exact DICOM folder structure.
        # This example assumes: RAW_DICOM_DIR/patient_id/RTPlan.dcm or similar.
        rtplan_search = os.path.join(RAW_DICOM_DIR, patient_id, "**", "*RP*.dcm")
        rtplan_matches = glob.glob(rtplan_search, recursive=True)
        
        if not rtplan_matches:
            print(f"Warning: Could not find RTPlan DICOM for {patient_id}. Skipping.")
            error_count += 1
            continue
            
        rtplan_path = rtplan_matches[0] # Take the first match
        
        # Output path for the 5th channel
        output_nifti_path = os.path.join(IMAGES_TR_DIR, f"{patient_id}_0004.nii.gz")
        
        # Skip if already generated (useful for resuming a broken batch process)
        if os.path.exists(output_nifti_path):
            print(f"[{patient_id}] Prior already exists. Skipping.")
            continue
            
        print(f"[{patient_id}] Generating beam prior...")
        try:
            generate_beam_prior(
                rtplan_path=rtplan_path, 
                ct_nifti_path=ct_path, 
                output_nifti_path=output_nifti_path,
                cylinder_radius_mm=50.0
            )
            print(f"  -> Saved to {output_nifti_path}")
            success_count += 1
        except Exception as e:
            print(f"  -> Error generating prior for {patient_id}: {e}")
            error_count += 1

    print(f"\nBatch processing complete.")
    print(f"Successfully generated: {success_count}")
    print(f"Errors/Skipped: {error_count}")

if __name__ == "__main__":
    main()
