"""
Audit all patients in 06_05_2026 folder.
For each patient folder, count DICOM files by modality.
Also analyzes RTSTRUCT ROI frequencies.
"""
# pyrefly: ignore [missing-import]
import pydicom
import os
import glob
from collections import Counter


# DATA_DIR = os.path.join(os.getcwd(), "Prostate prime d11 CT RT RP and RD")
DATA_DIR = os.path.join(os.getcwd(), "data/d12")
patient_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"Total patient folders: {len(patient_dirs)}")
print(f"{'Patient ID':<45} {'CT':>5} {'RTSTRUCT':>10} {'RTDOSE':>8} {'RTPLAN':>8} {'OTHER':>7}")
print("-" * 90)

total_files = 0
complete_patients = 0
missing_dose = []
missing_plan = []
missing_struct = []

for pid in patient_dirs:
    pdir = os.path.join(DATA_DIR, pid)
    dcm_files = glob.glob(os.path.join(pdir, "**", "*.dcm"), recursive=True)
    total_files += len(dcm_files)
    
    modality_counts = {}
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            mod = ds.Modality
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        except Exception:
            modality_counts["ERROR"] = modality_counts.get("ERROR", 0) + 1

    ct = modality_counts.get("CT", 0)
    rs = modality_counts.get("RTSTRUCT", 0)
    rd = modality_counts.get("RTDOSE", 0)
    rp = modality_counts.get("RTPLAN", 0)
    other_count = sum(v for k, v in modality_counts.items() if k not in ("CT", "RTSTRUCT", "RTDOSE", "RTPLAN"))
    
    # Track completeness
    has_all = ct > 0 and rs > 0 and rd > 0 and rp > 0
    if has_all:
        complete_patients += 1
    if rd == 0:
        missing_dose.append(pid)
    if rp == 0:
        missing_plan.append(pid)
    if rs == 0:
        missing_struct.append(pid)
    
    status = "✓" if has_all else "✗"
    print(f"{pid:<45} {ct:>5} {rs:>10} {rd:>8} {rp:>8} {other_count:>7}  {status}")

print("-" * 90)
print(f"\nSUMMARY:")
print(f"  Total patient folders:   {len(patient_dirs)}")
print(f"  Total DICOM files:       {total_files}")
print(f"  Complete (all 4 modalities): {complete_patients}")
print(f"  Missing RTDose:  {len(missing_dose)} -> {missing_dose[:5]}{'...' if len(missing_dose) > 5 else ''}")
print(f"  Missing RTPlan:  {len(missing_plan)} -> {missing_plan[:5]}{'...' if len(missing_plan) > 5 else ''}")
print(f"  Missing RTStruct: {len(missing_struct)} -> {missing_struct[:5]}{'...' if len(missing_struct) > 5 else ''}")


# Function to read RTSTRUCT ROIs
def read_rtstruct_roi_names(rtstruct_path):
    """
    Read RTSTRUCT DICOM and extract ROI names.
    
    Args:
        rtstruct_path: Path to RTSTRUCT DICOM file
        
    Returns:
        list: ROI names found in the structure set
    """
    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        if ds.Modality != "RTSTRUCT":
            return []
        
        roi_names = []
        if hasattr(ds, 'StructureSetROISequence'):
            for roi in ds.StructureSetROISequence:
                roi_name = getattr(roi, 'ROIName', 'Unknown')
                roi_names.append(roi_name)
        return roi_names
    except Exception as e:
        print(f"Error reading {rtstruct_path}: {e}")
        return []


def get_patient_rtstruct_path(patient_dir):
    """Find the RTSTRUCT file in a patient directory (searches recursively)."""
    dcm_files = glob.glob(os.path.join(patient_dir, "**/*.dcm"), recursive=True)
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if ds.Modality == "RTSTRUCT":
                return f
        except:
            continue
    return None


def create_roi_frequency_table():
    """
    Create ROI frequency table across all patients.
    Returns dict with ROI names and their occurrence counts.
    """
    all_roi_names = []
    patient_roi_count = {}
    
    print("\n" + "="*90)
    print("RTSTRUCT ROI ANALYSIS")
    print("="*90)
    
    for pid in patient_dirs:
        pdir = os.path.join(DATA_DIR, pid)
        rtstruct_path = get_patient_rtstruct_path(pdir)
        
        if rtstruct_path:
            roi_names = read_rtstruct_roi_names(rtstruct_path)
            patient_roi_count[pid] = len(roi_names)
            all_roi_names.extend(roi_names)
        else:
            patient_roi_count[pid] = 0
    
    # Count frequencies
    roi_counter = Counter(all_roi_names)
    
    print(f"\nTotal unique ROIs across all patients: {len(roi_counter)}")
    print(f"Total ROI instances: {sum(roi_counter.values())}")
    print(f"\n{'Rank':<6} {'ROI Name':<40} {'Count':>10}")
    print("-" * 58)
    
    sorted_rois = roi_counter.most_common()
    
    for rank, (roi_name, count) in enumerate(sorted_rois, 1):
        print(f"{rank:<6} {roi_name:<40} {count:>10}")
    
    return roi_counter, patient_roi_count


# Run ROI analysis
roi_freq, patient_counts = create_roi_frequency_table()


# Function to read RTDOSE file
def read_rtdose_dose(rtdose_path):
    """
    Read RTDOSE DICOM and extract dose information.
    
    Args:
        rtdose_path: Path to RTDOSE DICOM file
        
    Returns:
        dict: Dose information including total dose, fractions, etc.
    """
    try:
        ds = pydicom.dcmread(rtdose_path, stop_before_pixels=True)
        if ds.Modality != "RTDOSE":
            return None
        
        dose_info = {
            'path': rtdose_path,
            'dose_units': getattr(ds, 'DoseUnits', 'Unknown'),
            'dose_type': getattr(ds, 'DoseType', 'Unknown'),
            'dose_summation_type': getattr(ds, 'DoseSummationType', 'Unknown'),
        }
        
        # Get grid dimensions
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns') and hasattr(ds, 'NumberOfFrames'):
            dose_info['grid_size'] = (ds.Rows, ds.Columns, ds.NumberOfFrames)
        
        # Get dose scaling (needed to calculate actual dose values)
        if hasattr(ds, 'DoseGridScaling'):
            dose_info['dose_scaling'] = float(ds.DoseGridScaling)
        else:
            dose_info['dose_scaling'] = 1.0
            
        return dose_info
    except Exception as e:
        print(f"Error reading {rtdose_path}: {e}")
        return None


def get_patient_rtdose_path(patient_dir):
    """Find the RTDOSE file in a patient directory (searches recursively)."""
    dcm_files = glob.glob(os.path.join(patient_dir, "**/*.dcm"), recursive=True)
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if ds.Modality == "RTDOSE":
                return f
        except:
            continue
    return None


def create_dose_fraction_table():
    """
    Create table of total dose and fractions from RTDOSE files.
    """
    print("\n" + "="*90)
    print("RTDOSE DOSE ANALYSIS")
    print("="*90)
    
    dose_data = []
    
    for pid in patient_dirs:
        pdir = os.path.join(DATA_DIR, pid)
        rtdose_path = get_patient_rtdose_path(pdir)
        
        if rtdose_path:
            dose_info = read_rtdose_dose(rtdose_path)
            if dose_info:
                dose_data.append({
                    'patient_id': pid,
                    **dose_info
                })
    
    print(f"\nFound {len(dose_data)} RTDOSE files")
    print(f"\n{'Patient ID':<40} {'Dose Units':<12} {'Summation':<15} {'Grid Size':<25}")
    print("-" * 95)
    
    for d in dose_data:
        grid_str = str(d.get('grid_size', 'N/A')) if d.get('grid_size') else 'N/A'
        print(f"{d['patient_id']:<40} {d.get('dose_units', 'N/A'):<12} {d.get('dose_summation_type', 'N/A'):<15} {grid_str:<25}")
    
    return dose_data


# Run dose analysis
dose_info_list = create_dose_fraction_table()
