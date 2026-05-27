"""
utils/create_dummy_plan.py
===========================
Generates a minimal but TPS-compatible RTPLAN DICOM ("dummy plan") that
satisfies the mandatory ReferencedRTPlanSequence (300C,0002) requirement
in RTDOSE DICOMs when imported into clinical radiotherapy planning systems.

Static values  : institution, machine, beam geometry (single 6 MV dynamic field)
Dynamic values : patient identity, study UIDs, frame of reference, RTSTRUCT link,
                 and isocenter (computed from PTV centroid in the RTSTRUCT)

Usage
-----
    from utils.create_dummy_plan import create_dummy_plan_dicom

    plan_path = create_dummy_plan_dicom("/data/patient_001/dicom")
    print("Plan saved:", plan_path)

CLI
---
    python utils/create_dummy_plan.py --dicom-dir /data/patient_001/dicom
"""

import os
import re
import datetime
import argparse
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# ─────────────────────────────────────────────────────────────────────────────
# Static institution / machine constants
# (Values from the clinical template plan — change to match your centre)
# ─────────────────────────────────────────────────────────────────────────────

_INSTITUTION_NAME        = "Tata Medical Center"
_INSTITUTION_ADDRESS     = "14 MAR E-W \r\nACTION AREA 3\r\n700160\r\nNEW TOWN\r\nWEST BENGAL\r\nIndia"
_DEPT_NAME               = "Radiation Oncology"
_STATION_NAME            = "VMSTBOX171"
_MANUFACTURER            = "Varian Medical Systems"
_MANUFACTURER_MODEL      = "ARIA RadOnc"
_SOFTWARE_VERSIONS       = "17.0.0"
_DEVICE_SERIAL           = "879186791099"   # ARIA server serial

_MACHINE_NAME            = "TrueBeamSN3191"
_MACHINE_SERIAL          = "3191"
_SOURCE_AXIS_DIST        = "1000"
_PRIMARY_DOSIMETER_UNIT  = "MU"
_BEAM_TYPE               = "DYNAMIC"
_RADIATION_TYPE          = "PHOTON"
_DELIVERY_TYPE           = "TREATMENT"
_BEAM_ENERGY             = "6"
_DOSE_RATE               = "600"

# Control point 0
_GANTRY_ANGLE_START      = "179"
_GANTRY_DIR_START        = "CC"
_GANTRY_ANGLE_END        = "181"

_JAW_X                   = ["-50", "50"]
_JAW_Y                   = ["-50", "50"]

_TABLE_TOP_VERTICAL      = "100"
_TABLE_TOP_LONGITUDINAL  = "1200"
_TABLE_TOP_LATERAL       = "0"

_PATIENT_POSITION        = "HFS"
_PLAN_LABEL              = "Dummy Plan (AI)"
_PLAN_INTENT             = "CURATIVE"
_PLAN_GEOMETRY           = "PATIENT"
_APPROVAL_STATUS         = "UNAPPROVED"

_SERIES_NUMBER           = "11"

# ─────────────────────────────────────────────────────────────────────────────
# PTV name patterns  (identical to dicom_to_nnunet.py)
# ─────────────────────────────────────────────────────────────────────────────

_PTV_PATTERNS = [
    r"^CTVP$", r"^CTV_62", r"^PTV_62", r"^CTV62$", r"^PTV62$",
    r"^CTV_36", r"^PTV_36", r"^CTV 36", r"^PTV 36",
    r"^CTV_44", r"^PTV_44", r"^CTV_25", r"^PTV_25",
    r"^CTV 25", r"^PTV 25",
]


def _match_ptv_names(roi_names):
    matched = []
    for name in roi_names:
        for pat in _PTV_PATTERNS:
            if re.match(pat, name, re.IGNORECASE):
                matched.append(name)
                break
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# DICOM folder scanner
# ─────────────────────────────────────────────────────────────────────────────

def _scan_for_ct_and_rs(dicom_dir: str):
    """
    Return (ct_ref_ds, rs_ds) from the given folder (and its parent tree).
    ct_ref_ds : first CT slice DataSet (for patient/study metadata)
    rs_ds     : full RTSTRUCT DataSet (with ContourSequences for isocenter)
    """
    dicom_dir = Path(dicom_dir).resolve()
    ct_ref_ds = None
    rs_ds     = None

    for f in sorted(dicom_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            mod = getattr(ds, "Modality", "")
            if mod == "CT" and ct_ref_ds is None:
                ct_ref_ds = ds
            elif mod == "RTSTRUCT" and rs_ds is None:
                rs_ds = pydicom.dcmread(str(f))   # need full read for contours
        except Exception:
            continue
        if ct_ref_ds and rs_ds:
            break

    # Fallback: search parent tree for RTSTRUCT if not found locally
    if rs_ds is None:
        parent = dicom_dir.parent
        for f in sorted(parent.rglob("*.dcm")):
            if dicom_dir in f.parents:
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if getattr(ds, "Modality", "") == "RTSTRUCT":
                    rs_ds = pydicom.dcmread(str(f))
                    break
            except Exception:
                continue

    assert ct_ref_ds is not None, f"No CT DICOM found in: {dicom_dir}"
    assert rs_ds     is not None, f"No RTSTRUCT DICOM found in: {dicom_dir} or its parent."
    return ct_ref_ds, rs_ds


# ─────────────────────────────────────────────────────────────────────────────
# Isocenter computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ptv_centroid(rs_ds) -> list:
    """
    Compute the PTV centroid (patient coordinates, mm) from RTSTRUCT contour data.
    Falls back to [0, 0, 0] if no PTV contours are found.
    """
    roi_names = [roi.ROIName for roi in rs_ds.StructureSetROISequence]
    ptv_names = _match_ptv_names(roi_names)

    if not ptv_names:
        print("  [create_dummy_plan] WARNING: No PTV structure found — isocenter set to [0,0,0].")
        return [0.0, 0.0, 0.0]

    all_points = []
    for ptv_name in ptv_names:
        roi_number = None
        for roi in rs_ds.StructureSetROISequence:
            if roi.ROIName == ptv_name:
                roi_number = roi.ROINumber
                break
        if roi_number is None:
            continue
        for rc in rs_ds.ROIContourSequence:
            if rc.ReferencedROINumber != roi_number:
                continue
            if not hasattr(rc, "ContourSequence"):
                continue
            for contour in rc.ContourSequence:
                pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
                all_points.append(pts)

    if not all_points:
        print("  [create_dummy_plan] WARNING: PTV contours empty — isocenter set to [0,0,0].")
        return [0.0, 0.0, 0.0]

    centroid = np.vstack(all_points).mean(axis=0)
    print(f"  [create_dummy_plan] PTV centroid (isocenter): {centroid.tolist()}")
    return centroid.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# RTPLAN builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ds_item(**kwargs) -> Dataset:
    """Create a Dataset item and set attributes from kwargs."""
    item = Dataset()
    for k, v in kwargs.items():
        setattr(item, k, v)
    return item


def _build_tolerance_table() -> Dataset:
    bld_tol = Sequence([
        _ds_item(BeamLimitingDevicePositionTolerance="2", RTBeamLimitingDeviceType="X"),
        _ds_item(BeamLimitingDevicePositionTolerance="2", RTBeamLimitingDeviceType="ASYMX"),
        _ds_item(BeamLimitingDevicePositionTolerance="2", RTBeamLimitingDeviceType="Y"),
        _ds_item(BeamLimitingDevicePositionTolerance="1", RTBeamLimitingDeviceType="ASYMY"),
    ])
    tol = Dataset()
    tol.ToleranceTableNumber                   = "1"
    tol.ToleranceTableLabel                    = "Radical_TB"
    tol.GantryAngleTolerance                   = "0.1"
    tol.BeamLimitingDeviceAngleTolerance       = "0.1"
    tol.BeamLimitingDeviceToleranceSequence    = bld_tol
    tol.PatientSupportAngleTolerance           = "3"
    tol.TableTopPitchAngleTolerance            = 3.0
    tol.TableTopRollAngleTolerance             = 3.0
    tol.TableTopVerticalPositionTolerance      = "10"
    tol.TableTopLongitudinalPositionTolerance  = "50"
    tol.TableTopLateralPositionTolerance       = "10"
    return tol


def _build_beam_sequence(dose_ref_uid: str, isocenter: list) -> Dataset:
    # Beam limiting devices
    bld_seq = Sequence([
        _ds_item(RTBeamLimitingDeviceType="ASYMX", NumberOfLeafJawPairs="1"),
        _ds_item(RTBeamLimitingDeviceType="ASYMY", NumberOfLeafJawPairs="1"),
    ])

    # Control point 0 jaw positions
    jaw_seq = Sequence([
        _ds_item(RTBeamLimitingDeviceType="ASYMX", LeafJawPositions=_JAW_X),
        _ds_item(RTBeamLimitingDeviceType="ASYMY", LeafJawPositions=_JAW_Y),
    ])

    # Control point 0
    cp0 = Dataset()
    cp0.ControlPointIndex                  = "0"
    cp0.NominalBeamEnergy                  = _BEAM_ENERGY
    cp0.DoseRateSet                        = _DOSE_RATE
    cp0.BeamLimitingDevicePositionSequence = jaw_seq
    cp0.GantryAngle                        = _GANTRY_ANGLE_START
    cp0.GantryRotationDirection            = _GANTRY_DIR_START
    cp0.BeamLimitingDeviceAngle            = "0"
    cp0.BeamLimitingDeviceRotationDirection = "NONE"
    cp0.PatientSupportAngle                = "0"
    cp0.PatientSupportRotationDirection    = "NONE"
    cp0.TableTopEccentricAngle             = "0"
    cp0.TableTopEccentricRotationDirection = "NONE"
    cp0.TableTopVerticalPosition           = _TABLE_TOP_VERTICAL
    cp0.TableTopLongitudinalPosition       = _TABLE_TOP_LONGITUDINAL
    cp0.TableTopLateralPosition            = _TABLE_TOP_LATERAL
    cp0.IsocenterPosition                  = [f"{v:.10f}" for v in isocenter]
    cp0.CumulativeMetersetWeight           = "0"
    cp0.TableTopPitchAngle                 = 0.0
    cp0.TableTopPitchRotationDirection     = "NONE"
    cp0.TableTopRollAngle                  = 0.0
    cp0.TableTopRollRotationDirection      = "NONE"

    # Control point 1 (final)
    cp1 = Dataset()
    cp1.ControlPointIndex          = "1"
    cp1.GantryAngle                = _GANTRY_ANGLE_END
    cp1.GantryRotationDirection    = "NONE"
    cp1.CumulativeMetersetWeight   = "1"

    # Beam item
    beam = Dataset()
    beam.Manufacturer                       = _MANUFACTURER
    beam.InstitutionName                    = _INSTITUTION_NAME
    beam.InstitutionalDepartmentName        = _DEPT_NAME
    beam.ManufacturerModelName              = "TDS"
    beam.DeviceSerialNumber                 = _MACHINE_SERIAL
    beam.TreatmentMachineName               = _MACHINE_NAME
    beam.PrimaryDosimeterUnit               = _PRIMARY_DOSIMETER_UNIT
    beam.SourceAxisDistance                 = _SOURCE_AXIS_DIST
    beam.BeamLimitingDeviceSequence         = bld_seq
    beam.BeamNumber                         = "1"
    beam.BeamName                           = "Field 1"
    beam.BeamType                           = _BEAM_TYPE
    beam.RadiationType                      = _RADIATION_TYPE
    beam.TreatmentDeliveryType              = _DELIVERY_TYPE
    beam.NumberOfWedges                     = "0"
    beam.NumberOfCompensators               = "0"
    beam.NumberOfBoli                       = "0"
    beam.NumberOfBlocks                     = "0"
    beam.FinalCumulativeMetersetWeight      = "1"
    beam.NumberOfControlPoints              = "2"
    beam.ControlPointSequence               = Sequence([cp0, cp1])
    beam.ReferencedPatientSetupNumber       = "1"
    beam.ReferencedToleranceTableNumber     = "1"
    return beam


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_dummy_plan_dicom(
    dicom_dir:  str,
    overwrite:  bool = False,
) -> str:
    """
    Create a dummy RTPLAN DICOM and save it in *dicom_dir*.

    Parameters
    ----------
    dicom_dir : folder containing CT slices + RTSTRUCT .dcm
    overwrite : if False (default), skip creation if RP.dummy_plan.dcm already exists

    Returns
    -------
    str : absolute path to the saved RTPLAN DICOM
    """
    dicom_dir = str(Path(dicom_dir).resolve())
    out_path  = os.path.join(dicom_dir, "RP.dummy_plan.dcm")

    if not overwrite and os.path.exists(out_path):
        print(f"  [create_dummy_plan] Already exists, skipping: {out_path}")
        return out_path

    # ── 1. Gather metadata ───────────────────────────────────────────────────
    print("  [create_dummy_plan] Scanning DICOM folder for CT + RTSTRUCT...")
    ct_ref, rs_ds = _scan_for_ct_and_rs(dicom_dir)

    now       = datetime.datetime.now()
    date_str  = now.strftime("%Y%m%d")
    time_str  = now.strftime("%H%M%S.%f")
    plan_uid  = generate_uid()
    series_uid = generate_uid()
    dose_ref_uid = generate_uid()

    isocenter = _compute_ptv_centroid(rs_ds)

    # ── 2. File meta ─────────────────────────────────────────────────────────
    RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = RTPLAN_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = plan_uid
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID     = generate_uid()
    file_meta.ImplementationVersionName  = "AI_PLAN_1.0"

    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    # ── 3. Identification ────────────────────────────────────────────────────
    ds.SpecificCharacterSet    = "ISO_IR 192"
    ds.SOPClassUID             = RTPLAN_SOP_CLASS
    ds.SOPInstanceUID          = plan_uid
    ds.Modality                = "RTPLAN"
    ds.InstanceCreationDate    = date_str
    ds.InstanceCreationTime    = time_str
    ds.ContentDate             = date_str
    ds.ContentTime             = time_str

    # ── 4. Patient — copied from CT ──────────────────────────────────────────
    for tag in ["PatientName", "PatientID", "PatientBirthDate",
                "PatientBirthTime", "PatientSex"]:
        if hasattr(ct_ref, tag):
            setattr(ds, tag, getattr(ct_ref, tag))

    # ── 5. Study — MUST match CT ─────────────────────────────────────────────
    for tag in ["StudyInstanceUID", "StudyDate", "StudyTime", "StudyID",
                "AccessionNumber", "ReferringPhysicianName", "StudyDescription"]:
        if hasattr(ct_ref, tag):
            setattr(ds, tag, getattr(ct_ref, tag))

    # ── 6. Frame of Reference — MUST match CT ────────────────────────────────
    for tag in ["FrameOfReferenceUID", "PositionReferenceIndicator"]:
        if hasattr(ct_ref, tag):
            setattr(ds, tag, getattr(ct_ref, tag))

    # ── 7. Series ────────────────────────────────────────────────────────────
    ds.SeriesInstanceUID         = series_uid
    ds.SeriesNumber              = _SERIES_NUMBER
    ds.SeriesDescription         = _PLAN_LABEL
    ds.SeriesDate                = date_str
    ds.SeriesTime                = time_str
    ds.Manufacturer              = _MANUFACTURER
    ds.ManufacturerModelName     = _MANUFACTURER_MODEL
    ds.SoftwareVersions          = _SOFTWARE_VERSIONS
    ds.DeviceSerialNumber        = _DEVICE_SERIAL
    ds.InstitutionName           = _INSTITUTION_NAME
    ds.InstitutionAddress        = _INSTITUTION_ADDRESS
    ds.InstitutionalDepartmentName = _DEPT_NAME
    ds.StationName               = _STATION_NAME
    ds.OperatorsName             = ""

    # ── 8. RT Plan specific ──────────────────────────────────────────────────
    ds.RTPlanLabel               = _PLAN_LABEL
    ds.RTPlanDate                = date_str
    ds.RTPlanTime                = time_str
    ds.PlanIntent                = _PLAN_INTENT
    ds.RTPlanGeometry            = _PLAN_GEOMETRY
    ds.ApprovalStatus            = _APPROVAL_STATUS

    # ── 9. Dose Reference Sequence ───────────────────────────────────────────
    dose_ref_item = Dataset()
    dose_ref_item.DoseReferenceNumber         = "1"
    dose_ref_item.DoseReferenceUID            = dose_ref_uid
    dose_ref_item.DoseReferenceStructureType  = "SITE"
    dose_ref_item.DoseReferenceDescription    = "PTV PLAN1"
    dose_ref_item.DoseReferenceType           = "TARGET"
    ds.DoseReferenceSequence                  = Sequence([dose_ref_item])

    # ── 10. Tolerance Table ──────────────────────────────────────────────────
    ds.ToleranceTableSequence = Sequence([_build_tolerance_table()])

    # ── 11. Fraction Group Sequence ──────────────────────────────────────────
    ref_beam_item = Dataset()
    ref_beam_item.ReferencedDoseReferenceUID = dose_ref_uid
    ref_beam_item.ReferencedBeamNumber       = "1"

    frac_item = Dataset()
    frac_item.FractionGroupNumber               = "1"
    frac_item.NumberOfFractionsPlanned          = ""
    frac_item.NumberOfBeams                     = "1"
    frac_item.NumberOfBrachyApplicationSetups   = "0"
    frac_item.ReferencedBeamSequence            = Sequence([ref_beam_item])
    ds.FractionGroupSequence                    = Sequence([frac_item])

    # ── 12. Beam Sequence ────────────────────────────────────────────────────
    ds.BeamSequence = Sequence([_build_beam_sequence(dose_ref_uid, isocenter)])

    # ── 13. Patient Setup ────────────────────────────────────────────────────
    setup_item = Dataset()
    setup_item.PatientPosition    = _PATIENT_POSITION
    if hasattr(ct_ref, "PatientPosition"):
        setup_item.PatientPosition = ct_ref.PatientPosition
    setup_item.PatientSetupNumber = "1"
    setup_item.SetupTechnique     = "ISOCENTRIC"
    ds.PatientSetupSequence       = Sequence([setup_item])

    # ── 14. Referenced Structure Set ─────────────────────────────────────────
    rs_ref_item = Dataset()
    rs_ref_item.ReferencedSOPClassUID    = rs_ds.SOPClassUID
    rs_ref_item.ReferencedSOPInstanceUID = rs_ds.SOPInstanceUID
    ds.ReferencedStructureSetSequence    = Sequence([rs_ref_item])

    # ── 15. Write ────────────────────────────────────────────────────────────
    pydicom.dcmwrite(out_path, ds)
    print(f"  [create_dummy_plan] Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dummy RTPLAN DICOM from CT + RTSTRUCT metadata"
    )
    parser.add_argument("--dicom-dir", required=True,
                        help="Folder containing CT slices + RTSTRUCT .dcm")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing RP.dummy_plan.dcm")
    args = parser.parse_args()

    path = create_dummy_plan_dicom(args.dicom_dir, overwrite=args.overwrite)
    print(f"Done: {path}")
