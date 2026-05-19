import os
import glob
import pydicom
from imrt_beam import generate_beam_prior


def find_all_rtplans(dicom_root_dir, debug=False):
    """
    Recursively search entire DICOM directory for ALL RTPlan files.
    Returns a dict mapping dicom_folder_name -> rtplan_file_path.
    The folder name maps to prostate_XXX via alphabetical ordering.

    Args:
        dicom_root_dir: Root directory containing all patient DICOM folders
        debug: Print debug info

    Returns:
        dict: {dicom_folder_name: rtplan_path, ...}
    """
    if not os.path.exists(dicom_root_dir):
        print(f"    Dir does not exist: {dicom_root_dir}")
        return {}

    rtplans = {}

    # Get all patient directories (top level)
    patient_dirs = [d for d in os.listdir(dicom_root_dir)
                    if os.path.isdir(os.path.join(dicom_root_dir, d))]

    print(f"    Scanning {len(patient_dirs)} patient directories for RTPLAN...")

    for patient_dir in patient_dirs:
        patient_path = os.path.join(dicom_root_dir, patient_dir)

        # Walk this patient's directory tree
        for root, dirs, files in os.walk(patient_path):
            for f in files:
                file_path = os.path.join(root, f)

                # Skip obvious non-DICOM files
                if any(ext in file_path.lower() for ext in ['.txt', '.csv', '.json', '.md', '.py', '.nii', '.gz']):
                    continue

                try:
                    # Read only the header (stop before pixels) for speed
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                    if hasattr(ds, 'Modality') and ds.Modality == "RTPLAN":
                        rtplans[patient_dir] = file_path  # Key = folder name

                        if debug:
                            print(f"    -> RTPLAN in {patient_dir}: {os.path.basename(file_path)}")
                        break  # Found RTPLAN for this patient, move to next

                except Exception:
                    # Skip files that aren't valid DICOM
                    continue

    print(f"    Found {len(rtplans)} RTPLAN files total")
    return rtplans


# ==============================================================================
# CONFIGURATION - Set these paths to your local environment
# ==============================================================================
# Directory containing ALL DICOM files (mixed patients, flat or nested structure)
RAW_DICOM_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/data/Prostate PRIME Standard arm d69/"

# Directory containing your preprocessed NIfTI images for nnU-Net
IMAGES_TR_DIR = "./nnUNet_raw/Dataset001_ProstateDose/imagesTr"


def build_patient_mapping(dicom_root_dir):
    """
    Build mapping from DICOM folder name to prostate_XXX ID.
    Matches the logic in dicom_to_nnunet.py: alphabetical order -> sequential IDs.

    Returns dict: {dicom_folder_name: "prostate_XXX"}
    """
    patient_dirs = sorted([d for d in os.listdir(dicom_root_dir)
                           if os.path.isdir(os.path.join(dicom_root_dir, d))])

    mapping = {}
    for case_id, pdir in enumerate(patient_dirs):
        prostate_id = f"prostate_{case_id:03d}"
        mapping[pdir] = prostate_id

    return mapping


def main():
    print(f"Starting batch generation of IMRT beam priors...")
    print(f"DICOM source: {RAW_DICOM_DIR}")
    print(f"NIfTI output: {IMAGES_TR_DIR}")

    # Build patient ID mapping (DICOM folder -> prostate_XXX)
    print("\n[0/3] Building patient ID mapping...")
    patient_mapping = build_patient_mapping(RAW_DICOM_DIR)
    print(f"  Found {len(patient_mapping)} patients in DICOM directory")

    # Find ALL RTPLAN files and map to prostate_XXX
    print("\n[1/3] Scanning for all RTPLAN files...")
    rtplan_by_dicom_id = find_all_rtplans(RAW_DICOM_DIR, debug=False)

    if not rtplan_by_dicom_id:
        print("No RTPLAN files found!")
        return

    # Convert to prostate_XXX keys
    rtplan_map = {}
    for dicom_folder, rtplan_path in rtplan_by_dicom_id.items():
        prostate_id = patient_mapping.get(dicom_folder)
        if prostate_id:
            rtplan_map[prostate_id] = rtplan_path

    print(f"\nFound {len(rtplan_map)} RTPLAN files (mapped to prostate_XXX)")

    # Find all CT NIfTI files
    print("\n[2/3] Finding CT NIfTI files...")
    ct_files = sorted(glob.glob(os.path.join(IMAGES_TR_DIR, "*_0000.nii.gz")))

    if not ct_files:
        print(f"No CT files found in {IMAGES_TR_DIR}")
        return

    print(f"Found {len(ct_files)} CT NIfTI files")

    # Match and process
    print("\n[3/3] Generating beam priors...")
    success_count = 0
    error_count = 0
    skipped_count = 0

    for ct_path in ct_files:
        # Extract patient ID from filename (e.g., 'prostate_001' from 'prostate_001_0000.nii.gz')
        basename = os.path.basename(ct_path)
        patient_id = basename.replace("_0000.nii.gz", "")

        # Check if we have an RTPLAN for this patient
        rtplan_path = rtplan_map.get(patient_id)

        if rtplan_path is None:
            # Try alternative patient ID formats
            # DICOM PatientID might be different from our naming
            found = False
            for pid, path in rtplan_map.items():
                if patient_id in pid or pid in patient_id:
                    rtplan_path = path
                    found = True
                    if success_count < 3:  # Debug first few
                        print(f"  Matched {patient_id} -> DICOM PatientID: {pid}")
                    break

            if not found:
                print(f"[{patient_id}] No matching RTPLAN found. Skipping.")
                error_count += 1
                continue

        # Output path for the 5th channel
        output_nifti_path = os.path.join(IMAGES_TR_DIR, f"{patient_id}_0004.nii.gz")

        # Skip if already generated
        if os.path.exists(output_nifti_path):
            skipped_count += 1
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
            print(f"  -> Error: {e}")
            error_count += 1

    print(f"\n{'='*50}")
    print(f"Batch processing complete.")
    print(f"Successfully generated: {success_count}")
    print(f"Already existed (skipped): {skipped_count}")
    print(f"Errors/No RTPLAN: {error_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
