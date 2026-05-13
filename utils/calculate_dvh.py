"""
DVH (Dose Volume Histogram) Calculator using SimpleITK and numpy
Calculates DVH metrics for each patient and all ROIs.

This implementation manually handles coordinate alignment between RTDOSE and RTSTRUCT,
which is more reliable than open-source libraries that have coordinate system bugs.
"""
import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import glob
from collections import defaultdict
from dotenv import load_dotenv


def find_patient_folders(root_dir):
    """Find all patient subfolders in root directory."""
    patient_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            patient_dirs.append(item_path)
    return sorted(patient_dirs)


def find_dicom_files(patient_dir):
    """Find CT, RTSTRUCT, RTPLAN, and RTDOSE files in patient directory."""
    dcm_files = glob.glob(os.path.join(patient_dir, "**/*.dcm"), recursive=True)
    
    ct_files = []
    rtstruct_file = None
    rtplan_file = None
    rtdose_file = None
    
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            modality = ds.Modality
            if modality == "CT":
                ct_files.append(f)
            elif modality == "RTSTRUCT":
                rtstruct_file = f
            elif modality == "RTPLAN":
                rtplan_file = f
            elif modality == "RTDOSE":
                rtdose_file = f
        except:
            continue
    
    return ct_files, rtstruct_file, rtplan_file, rtdose_file


def read_rtplan_info(rtplan_file):
    """
    Read prescribed dose and number of fractions from RTPLAN file.
    Returns tuple (prescribed_dose_gy, num_fractions) or (None, None) if not found.
    """
    if not rtplan_file:
        return None, None
    
    try:
        ds = pydicom.dcmread(rtplan_file, stop_before_pixels=True)
        
        prescribed_dose = None
        num_fractions = None
        
        # Try to get dose from DoseReferenceSequence
        if hasattr(ds, 'DoseReferenceSequence'):
            for dose_ref in ds.DoseReferenceSequence:
                dose_type = getattr(dose_ref, 'DoseReferenceType', None)
                if dose_type == 'TARGET':
                    if hasattr(dose_ref, 'TargetPrescriptionDose'):
                        prescribed_dose = float(dose_ref.TargetPrescriptionDose)
        
        # Try FractionGroupSequence for dose and fractions
        if hasattr(ds, 'FractionGroupSequence'):
            for fg in ds.FractionGroupSequence:
                # Get number of fractions
                if hasattr(fg, 'NumberOfFractionsPlanned'):
                    num_fractions = int(fg.NumberOfFractionsPlanned)
                
                # Get prescribed dose
                if hasattr(fg, 'ReferencedDoseReferenceSequence'):
                    for rdr in fg.ReferencedDoseReferenceSequence:
                        if hasattr(rdr, 'TargetPrescriptionDose'):
                            prescribed_dose = float(rdr.TargetPrescriptionDose)
        
        return prescribed_dose, num_fractions
    except Exception as e:
        print(f"    Warning: Could not read RTPLAN info: {e}")
        return None, None


def load_ct_volume(ct_files):
    """Load CT volume from DICOM series."""
    if not ct_files:
        return None
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(os.path.dirname(ct_files[0]))
    if series_ids:
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.dirname(ct_files[0]), series_ids[0])
        reader.SetFileNames(dicom_names)
        ct_image = reader.Execute()
        
        # Handle 4D CT (some DICOM have extra dimension with size 1)
        ct_size = ct_image.GetSize()
        if len(ct_size) == 4 and ct_size[3] == 1:
            ct_image = ct_image[:,:,:,0]  # Extract 3D
            print(f"    Converted 4D CT to 3D")
        
        # Fix direction matrix if it's 4x4 (16 elements instead of 3x3 = 9 elements)
        direction = ct_image.GetDirection()
        if len(direction) == 16:
            # Extract 3x3 from 4x4 matrix (row-major order)
            new_direction = (
                direction[0], direction[1], direction[2],
                direction[4], direction[5], direction[6],
                direction[8], direction[9], direction[10]
            )
            ct_image.SetDirection(new_direction)
            print(f"    Fixed 4x4 direction matrix")
        
        return ct_image
    return None


def load_rtdose(rtdose_file):
    """Load RTDOSE as SimpleITK image."""
    ds = pydicom.dcmread(rtdose_file)
    dose_array = ds.pixel_array * float(getattr(ds, 'DoseGridScaling', 1.0))
    
    # Handle 4D dose arrays (some RTDOSE files have extra dimension)
    if len(dose_array.shape) == 4:
        dose_array = dose_array[0]
    
    # Create SimpleITK image
    dose_image = sitk.GetImageFromArray(dose_array)
    
    # Set spacing from DICOM
    if hasattr(ds, 'GridFrameOffsetVector') and hasattr(ds, 'ImagePositionPatient'):
        ipp = ds.ImagePositionPatient
        # PixelSpacing is [row_spacing, col_spacing] = [y, x]
        if hasattr(ds, 'PixelSpacing'):
            spacing = [
                float(ds.PixelSpacing[1]),  # x spacing (column)
                float(ds.PixelSpacing[0]),  # y spacing (row)
                abs(float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])) if len(ds.GridFrameOffsetVector) > 1 else 1.0
            ]
        else:
            spacing = [1.0, 1.0, 1.0]
        
        dose_image.SetSpacing(spacing)
        dose_image.SetOrigin([float(ipp[0]), float(ipp[1]), float(ipp[2])])
        
        # Set direction cosines from ImageOrientationPatient if available
        if hasattr(ds, 'ImageOrientationPatient'):
            iop = ds.ImageOrientationPatient
            x_cos = [float(iop[0]), float(iop[1]), float(iop[2])]
            y_cos = [float(iop[3]), float(iop[4]), float(iop[5])]
            z_cos = [
                x_cos[1]*y_cos[2] - x_cos[2]*y_cos[1],
                x_cos[2]*y_cos[0] - x_cos[0]*y_cos[2],
                x_cos[0]*y_cos[1] - x_cos[1]*y_cos[0]
            ]
            direction = x_cos + y_cos + z_cos
            dose_image.SetDirection(direction)
    
    return dose_image, ds


def read_rtstruct_rois(rtstruct_file):
    """Read ROI names from RTSTRUCT."""
    ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
    roi_names = []
    if hasattr(ds, 'StructureSetROISequence'):
        for roi in ds.StructureSetROISequence:
            roi_names.append(getattr(roi, 'ROIName', 'Unknown'))
    return roi_names, ds


def rtstruct_to_masks(rs_ds, ct_image):
    """
    Convert RTStruct contours to 3D binary masks aligned to CT geometry.
    Returns a dictionary of {roi_name: mask_array}
    """
    ref_size = ct_image.GetSize()
    ref_origin = ct_image.GetOrigin()
    ref_spacing = ct_image.GetSpacing()
    ref_direction = ct_image.GetDirection()
    
    # Get physical to index transform
    masks = {}
    
    if not hasattr(rs_ds, 'StructureSetROISequence') or not hasattr(rs_ds, 'ROIContourSequence'):
        return masks
    
    # Create a mapping from ROI number to ROI name
    roi_number_to_name = {}
    for roi in rs_ds.StructureSetROISequence:
        roi_number_to_name[roi.ROINumber] = getattr(roi, 'ROIName', 'Unknown')
    
    # Process each ROI contour
    for contour in rs_ds.ROIContourSequence:
        roi_number = contour.ReferencedROINumber
        roi_name = roi_number_to_name.get(roi_number, f'ROI_{roi_number}')
        
        if not hasattr(contour, 'ContourSequence'):
            continue
        
        # Initialize mask
        mask = np.zeros(ref_size[::-1], dtype=np.uint8)  # SimpleITK uses [x,y,z], numpy uses [z,y,x]
        
        # Process each contour slice
        for contour_item in contour.ContourSequence:
            if not hasattr(contour_item, 'ContourData'):
                continue
            
            # Get contour points (x, y, z triplets)
            points = np.array(contour_item.ContourData).reshape(-1, 3)
            
            if len(points) < 3:
                continue
            
            # Convert physical coordinates to voxel indices using CT geometry
            # Use SimpleITK's transform
            for i in range(len(points)):
                point = points[i]
                # Transform from physical to index coordinates
                index = ct_image.TransformPhysicalPointToIndex((float(point[0]), float(point[1]), float(point[2])))
                
                if 0 <= index[2] < ref_size[2]:
                    # For this slice, we'll use the contour points to create a polygon
                    pass
            
            # Simple approach: for each point, mark the nearest voxel
            z_idx = None
            x_indices = []
            y_indices = []
            
            for point in points:
                index = ct_image.TransformPhysicalPointToIndex((float(point[0]), float(point[1]), float(point[2])))
                
                if 0 <= index[0] < ref_size[0] and 0 <= index[1] < ref_size[1] and 0 <= index[2] < ref_size[2]:
                    x_indices.append(index[0])
                    y_indices.append(index[1])
                    z_idx = index[2]
            
            if z_idx is not None and len(x_indices) >= 3:
                # Fill polygon on this slice
                from skimage.draw import polygon
                rr, cc = polygon(y_indices, x_indices, shape=(ref_size[1], ref_size[0]))
                mask[z_idx, rr, cc] = 1
        
        masks[roi_name] = mask
    
    return masks


def resample_dose_to_ct(dose_image, ct_image):
    """Resample dose grid to match CT geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampled_dose = resampler.Execute(dose_image)
    return resampled_dose


def calculate_dvh_metrics(dose_array, mask_array, spacing):
    """
    Calculate DVH metrics for a structure.
    
    Returns:
        dict with volume_cc, mean_dose_gy, min_dose_gy, max_dose_gy
    """
    # Calculate voxel volume in cc (mm^3 to cc)
    voxel_volume_cc = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    
    # Get dose values within mask
    structure_dose = dose_array[mask_array > 0]
    
    if len(structure_dose) == 0:
        return {
            'volume_cc': 0.0,
            'mean_dose_gy': 0.0,
            'min_dose_gy': 0.0,
            'max_dose_gy': 0.0
        }
    
    # Calculate metrics
    volume_cc = len(structure_dose) * voxel_volume_cc
    mean_dose = np.mean(structure_dose)
    min_dose = np.min(structure_dose)
    max_dose = np.max(structure_dose)
    
    return {
        'volume_cc': round(volume_cc, 2),
        'mean_dose_gy': round(mean_dose, 2),
        'min_dose_gy': round(min_dose, 2),
        'max_dose_gy': round(max_dose, 2)
    }


def process_patient(patient_dir):
    """Process a single patient and return DVH data."""
    patient_name = os.path.basename(patient_dir)
    print(f"\nProcessing: {patient_name}")
    
    # Find DICOM files
    ct_files, rtstruct_file, rtplan_file, rtdose_file = find_dicom_files(patient_dir)
    
    if not ct_files:
        print(f"  ERROR: No CT files found")
        return []
    if not rtstruct_file:
        print(f"  ERROR: No RTSTRUCT file found")
        return []
    if not rtdose_file:
        print(f"  ERROR: No RTDOSE file found")
        return []
    
    print(f"  CT: {len(ct_files)} slices, RTSTRUCT: {os.path.basename(rtstruct_file)}, RTDOSE: {os.path.basename(rtdose_file)}")
    
    # Read prescribed dose and fractions from RTPLAN
    prescribed_dose = None
    num_fractions = None
    if rtplan_file:
        print(f"  RTPLAN: {os.path.basename(rtplan_file)}")
        prescribed_dose, num_fractions = read_rtplan_info(rtplan_file)
        if prescribed_dose:
            print(f"    Prescribed dose: {prescribed_dose} Gy")
        if num_fractions:
            print(f"    Number of fractions: {num_fractions}")
    
    try:
        # Load CT volume
        ct_image = load_ct_volume(ct_files)
        if ct_image is None:
            print(f"  ERROR: Failed to load CT")
            return []
        
        print(f"    CT loaded: {ct_image.GetSize()}")
        
        # Load dose
        dose_image, dose_ds = load_rtdose(rtdose_file)
        print(f"    Dose loaded: {dose_image.GetSize()}")
        
        # Resample dose to match CT
        print(f"  Resampling dose to CT geometry...")
        resampled_dose = resample_dose_to_ct(dose_image, ct_image)
        dose_array = sitk.GetArrayFromImage(resampled_dose)
        ct_spacing = ct_image.GetSpacing()
        
        # Read RTSTRUCT and create masks
        print(f"  Loading RTSTRUCT...")
        roi_names, rs_ds = read_rtstruct_rois(rtstruct_file)
        print(f"    Found {len(roi_names)} ROIs")
        
        # Convert contours to masks
        print(f"  Converting contours to masks...")
        masks = rtstruct_to_masks(rs_ds, ct_image)
        print(f"    Created {len(masks)} masks")
        
        # Calculate DVH for all structures
        dvh_results = []
        print(f"\n  Calculating DVH for all structures...")
        
        for roi_name in roi_names:
            if roi_name in masks:
                mask_array = masks[roi_name]
                metrics = calculate_dvh_metrics(dose_array, mask_array, ct_spacing)
                
                result = {
                    'patient': patient_name,
                    'roi': roi_name,
                    'prescribed_dose_gy': round(prescribed_dose, 2) if prescribed_dose else None,
                    'num_fractions': num_fractions,
                    'volume_cc': metrics['volume_cc'],
                    'mean_dose_gy': metrics['mean_dose_gy'],
                    'min_dose_gy': metrics['min_dose_gy'],
                    'max_dose_gy': metrics['max_dose_gy']
                }
                dvh_results.append(result)
                
                if metrics['volume_cc'] > 0:
                    print(f"    {roi_name}: Vol={metrics['volume_cc']}cc, Mean={metrics['mean_dose_gy']}Gy")
            else:
                print(f"    {roi_name}: ✗ (no mask created)")
        
        return dvh_results
        
    except Exception as e:
        print(f"  ERROR processing patient: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    # Configuration
    # Get project root (parent of utils directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load environment variables from .env file (REQUIRED)
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        print(f"ERROR: Config file not found at {env_path}")
        print("Please create a .env file with DATA_DIR variable")
        return
    
    load_dotenv(env_path)
    print(f"Loaded configuration from: {env_path}")
    
    # Get data directory from environment (must be set in .env file)
    DATA_DIR = os.getenv("DATA_DIR")
    
    if not DATA_DIR:
        print("ERROR: DATA_DIR not set in .env file")
        print("Add to .env: DATA_DIR=/full/path/to/patient/data")
        return
    
    # Output directory is static: project_root/data/dvh
    OUTPUT_DIR = os.path.join(project_root, "data", "dvh")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Add timestamp to output filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"dvh_results_{timestamp}.csv")
    
    print("="*80)
    print("DVH CALCULATOR (SimpleITK + numpy implementation)")
    print("="*80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output file: {OUTPUT_CSV}")
    
    # Find all patient folders
    patient_folders = find_patient_folders(DATA_DIR)
    print(f"\nFound {len(patient_folders)} patient folders")
    
    # Process all patients
    all_results = []
    for patient_dir in patient_folders:
        results = process_patient(patient_dir)
        all_results.extend(results)
    
    # Create DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED TO: {OUTPUT_CSV}")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(f"  Total patients: {df['patient'].nunique()}")
        print(f"  Total structures: {len(df)}")
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string())
    else:
        print("\nNo results generated (check if DICOM files exist)")


if __name__ == "__main__":
    main()
