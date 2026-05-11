"""
Audit all patients in 06_05_2026 folder.
For each patient folder, count DICOM files by modality.
"""
# pyrefly: ignore [missing-import]
import pydicom
import os
import glob


DATA_DIR = os.path.join(os.getcwd(), "Prostate prime d11 CT RT RP and RD")
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
