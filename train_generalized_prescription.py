"""
train_generalized_prescription.py
=================================
Generalized prescription training - supports multiple prescription schemes:
- 62 Gy, 36.5 Gy, 36.25 Gy, 35 Gy (and original 75 Gy SIB)

Key changes from train_dummy_physics.py:
1. Prescription dose passed as metadata per patient
2. Constraint files loaded per prescription scheme
3. Model receives prescription as 8th input channel (constant)
4. Loss normalization uses per-patient prescription
"""

import os
import sys
import re
import csv
import math
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from glob import glob

import pydicom
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ConcatItemsd, ToTensord, DeleteItemsd,
    MapTransform, RandCropByLabelClassesd,
)
from monai.data import Dataset, list_data_collate
from monai.data import PersistentDataset
from monai.inferers import sliding_window_inference

# ===================================================================
# Configuration
# ===================================================================

DATA_DIR = os.path.join(os.getcwd(), "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

CHANNELS = ["0000", "0001", "0002", "0003", "0004", "0005", "0006"]  # 7 base channels
TARGET_SPACING = (1.27, 1.27, 2.5)
PATCH_SIZE = (128, 128, 64)

# Prescription scheme mapping
PRESCRIPTION_SCHEMES = {
    "prostate_62gy": 62.0,
    "prostate_36.5gy": 36.5,
    "prostate_36.25gy": 36.25,
    "prostate_35gy": 35.0,
    "prostate_75gy_sib": 75.0,  # Original SIB scheme
}

# Default constraint files per scheme
DEFAULT_CONSTRAINT_FILES = {
    "prostate_62gy": "./prostate_62gy_constraints.csv",
    "prostate_36.5gy": "./prostate_36.5gy_constraints.csv",
    "prostate_36.25gy": "./prostate_36.25gy_constraints.csv",
    "prostate_35gy": "./prostate_35gy_constraints.csv",
    "prostate_75gy_sib": "./prostate_prime_constraints_v3.csv",
}

# Patient metadata file - maps patient_id to prescription scheme
PATIENT_METADATA_FILE = os.path.join(DATA_DIR, "patient_prescriptions.json")

GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
VAL_EVERY_N_EPOCHS = int(os.environ.get("VAL_EVERY_N_EPOCHS", "1"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "30"))

# SIB canonical keys (for patients with simultaneous integrated boost)
SIB_KEYS = ["PTV60", "PTV55", "PTV54", "PTV44", "PTV36", "PTV25", "PTV62"]
SIB_ORDER = [
    ("PTV25", 25.0), ("PTV36", 36.0), ("PTV44", 44.0),
    ("PTV54", 54.0), ("PTV55", 55.0), ("PTV60", 60.0), ("PTV62", 62.0),
]

ALL_KEYS = [f"ch_{i}" for i in range(7)] + ["dose_label", "ring_mask", "bowel_mask", "femur_mask"] + SIB_KEYS


# ===================================================================
# Patient Metadata Management
# ===================================================================

def load_patient_metadata():
    """Load patient_id -> prescription_scheme mapping."""
    if os.path.exists(PATIENT_METADATA_FILE):
        with open(PATIENT_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_patient_metadata(metadata):
    """Save patient_id -> prescription_scheme mapping."""
    with open(PATIENT_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def infer_prescription_from_contours(patient_id, rs_file):
    """
    Infer prescription scheme from RTSTRUCT contour names.
    Returns prescription_scheme string or None.
    """
    try:
        ds = pydicom.dcmread(rs_file, force=True)
        roi_names = [roi.ROIName for roi in ds.StructureSetROISequence]
        
        # Check for PTV dose levels
        has_ptv62 = any(re.search(r'PTV.*62|CTV.*62', name, re.I) for name in roi_names)
        has_ptv60 = any(re.search(r'PTV.*60|CTV.*60', name, re.I) for name in roi_names)
        has_ptv44 = any(re.search(r'PTV.*44|CTV.*44', name, re.I) for name in roi_names)
        has_ptv36 = any(re.search(r'PTV.*36|CTV.*36', name, re.I) for name in roi_names)
        has_ptv35 = any(re.search(r'PTV.*35|CTV.*35', name, re.I) for name in roi_names)
        
        if has_ptv62:
            return "prostate_62gy"
        elif has_ptv60 and has_ptv44:
            return "prostate_75gy_sib"  # Original SIB scheme
        elif has_ptv36:
            # Could be 36.5 or 36.25 - check RTPLAN if available
            return "prostate_36.5gy"  # Default
        elif has_ptv35:
            return "prostate_35gy"
        
        return "prostate_75gy_sib"  # Default fallback
    except Exception as e:
        print(f"Warning: Could not infer prescription for {patient_id}: {e}")
        return "prostate_75gy_sib"


def assign_prescriptions_to_patients():
    """
    Scan dataset and assign prescription schemes to patients.
    Creates patient_prescriptions.json if it doesn't exist.
    """
    metadata = load_patient_metadata()
    
    # Find all patients
    patient_ids = set()
    for f in os.listdir(IMAGES_DIR):
        if f.endswith('_0000.nii.gz'):
            patient_id = f.replace('_0000.nii.gz', '')
            patient_ids.add(patient_id)
    
    # Find RTSTRUCT files
    dicom_dirs = glob(os.path.join(DATA_DIR, "../../testdata/*/"))
    
    new_assignments = 0
    for patient_id in patient_ids:
        if patient_id in metadata:
            continue  # Already assigned
        
        # Try to find DICOM for this patient
        prescription = "prostate_75gy_sib"  # Default
        
        # Look for RTSTRUCT in common locations
        for dicom_dir in dicom_dirs:
            for root, dirs, files in os.walk(dicom_dir):
                for f in files:
                    if 'RTSTRUCT' in f or f.endswith('-RS.dcm'):
                        try:
                            inferred = infer_prescription_from_contours(patient_id, os.path.join(root, f))
                            if inferred:
                                prescription = inferred
                                break
                        except:
                            pass
                if prescription != "prostate_75gy_sib":
                    break
        
        metadata[patient_id] = prescription
        new_assignments += 1
    
    save_patient_metadata(metadata)
    print(f"Assigned prescriptions to {new_assignments} patients")
    print(f"Total patients in metadata: {len(metadata)}")
    
    # Print distribution
    scheme_counts = {}
    for scheme in metadata.values():
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1
    print("Prescription distribution:", scheme_counts)
    
    return metadata


# ===================================================================
# Constraint Loading (Prescription-Aware)
# ===================================================================

def load_clinical_constraints_for_prescription(prescription_scheme, patient_class="N0"):
    """
    Load constraints appropriate for the prescription scheme.
    """
    csv_path = DEFAULT_CONSTRAINT_FILES.get(prescription_scheme, 
                                             DEFAULT_CONSTRAINT_FILES["prostate_75gy_sib"])
    
    prescription_gy = PRESCRIPTION_SCHEMES.get(prescription_scheme, 75.0)
    
    return load_clinical_constraints(csv_path, prescription_gy, patient_class)


def load_clinical_constraints(csv_path, prescription_gy, patient_class="N0"):
    """
    Parse constraint CSV into structured dicts with prescription-aware normalization.
    """
    v_accum = {}
    ptv_coverage = []
    ptv_max_dose_gy = None
    nplus_suffix = "_Nplus"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"].strip()
            struct_type = row["Type"].strip()
            ctype = row["Constraint_Type"].strip()
            c_val_raw = row["Constraint_Value"].strip()
            c_unit = row["Constraint_Unit"].strip()
            obj_val_raw = row["Objective_Value"].strip()
            obj_unit = row["Objective_Unit"].strip()
            obj_type = row["Objective_Type"].strip()

            if not obj_val_raw:
                continue

            is_nplus = name.endswith(nplus_suffix)
            if patient_class == "N0" and is_nplus:
                continue
            if patient_class == "N+" and not is_nplus:
                continue

            canonical = name[: -len(nplus_suffix)] if is_nplus else name

            # V-Type constraints
            if ctype == "V" and canonical in ("Bladder", "Anorectum", "PenileBulb", "Bag_Bowel"):
                if obj_unit.strip() != "%":
                    continue
                dose_thresh_gy = float(c_val_raw)
                obj_value = float(obj_val_raw)

                key = (canonical, dose_thresh_gy)
                if key not in v_accum:
                    v_accum[key] = {"optimal_v": float("nan"), "mandatory_v": float("nan")}

                if obj_type == "Optimal":
                    v_accum[key]["optimal_v"] = obj_value
                elif obj_type == "Mandatory":
                    v_accum[key]["mandatory_v"] = obj_value

            # D-Type: PTV max dose
            if (ctype == "D" and struct_type == "PTV"
                    and c_val_raw == "Max" and c_unit == "Gy"):
                if obj_type == "Mandatory":
                    ptv_max_dose_gy = float(obj_val_raw)

            # D-Type: PTV coverage
            if (ctype == "D" and struct_type == "PTV"
                    and c_unit == "%" and obj_unit == "%"):
                try:
                    percentile = float(c_val_raw)
                except ValueError:
                    continue
                if percentile >= 90 and obj_type == "Mandatory":
                    ptv_coverage.append({"metric": f"D{int(percentile)}", "fraction": float(obj_val_raw)})

    # Build v_constraints with prescription-aware normalization
    v_constraints = {"Bladder": [], "Anorectum": [], "PenileBulb": [], "Bag_Bowel": []}
    for (organ, dose_gy), tiers in sorted(v_accum.items(), key=lambda x: x[0][1]):
        if math.isnan(tiers["mandatory_v"]):
            continue
        norm_dose = dose_gy / prescription_gy  # Use actual prescription
        v_constraints[organ].append({
            "dose_gy": dose_gy,
            "norm_dose": norm_dose,
            "optimal_v": tiers["optimal_v"],
            "mandatory_v": tiers["mandatory_v"],
        })

    return {
        "v_type": v_constraints,
        "d_type": {
            "PTV_max_dose_gy": ptv_max_dose_gy,
            "PTV_coverage": ptv_coverage,
        },
        "prescription_gy": prescription_gy,
    }


# ===================================================================
# Prescription-Aware Loss Function
# ===================================================================

class GeneralizedPhysicsDoseLoss(nn.Module):
    """
    Physics-guided loss that accepts prescription_gy as a parameter.
    Normalizes all thresholds by the per-sample prescription.
    """

    def __init__(
        self,
        lambda_mse=25.0,
        lambda_optimal=0.0,
        lambda_mandatory=0.0,
        lambda_ptv=0.0,
        lambda_ring=0.0,
        lambda_smooth=0.0,
        lambda_laplacian=0.0,
        lambda_anticollapse=0.0,
        lambda_ptv_max=0.0,
        lambda_homogeneity=0.0,
        lambda_global_ceil=2.0,
        lambda_shell_inner=0.0,
        lambda_shell_outer=0.0,
        lambda_bowel=0.0,
        lambda_femur=0.0,
        lambda_penile=0.0,
        lambda_bg=0.0,
        k_steepness=50.0,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.lambda_optimal = lambda_optimal
        self.lambda_mandatory = lambda_mandatory
        self.lambda_ptv = lambda_ptv
        self.lambda_ring = lambda_ring
        self.lambda_smooth = lambda_smooth
        self.lambda_laplacian = lambda_laplacian
        self.lambda_anticollapse = lambda_anticollapse
        self.lambda_ptv_max = lambda_ptv_max
        self.lambda_homogeneity = lambda_homogeneity
        self.lambda_global_ceil = lambda_global_ceil
        self.lambda_shell_inner = lambda_shell_inner
        self.lambda_shell_outer = lambda_shell_outer
        self.lambda_bowel = lambda_bowel
        self.lambda_femur = lambda_femur
        self.lambda_penile = lambda_penile
        self.lambda_bg = lambda_bg
        self.lambda_body = 20.0
        self.k = k_steepness

    def calculate_dvh_volume(self, predicted_dose, organ_mask, norm_dose_threshold):
        """Differentiable DVH volume calculation."""
        organ_voxels = predicted_dose * organ_mask
        n_organ = organ_mask.sum()
        if n_organ < 1.0:
            return torch.tensor(0.0, device=predicted_dose.device, dtype=predicted_dose.dtype)
        step_approx = torch.sigmoid(self.k * (organ_voxels - norm_dose_threshold))
        step_approx = step_approx * organ_mask
        volume_fraction = step_approx.sum() / n_organ
        return volume_fraction

    def forward(self, pred_dose, true_dose, bladder_mask, rectum_mask,
                ptv_mask, ring_mask, inputs, bowel_mask, femur_mask, bag_bowel_mask,
                constraints_dict, prescription_gy):
        """
        Parameters:
        -----------
        pred_dose, true_dose: normalized by prescription_gy
        constraints_dict: loaded constraints for this prescription
        prescription_gy: scalar tensor or float, the prescription dose
        """
        # Ensure prescription_gy is a tensor
        if not isinstance(prescription_gy, torch.Tensor):
            prescription_gy = torch.tensor(prescription_gy, device=pred_dose.device, dtype=pred_dose.dtype)
        
        # ------ 1. L_MSE (dose-weighted) ------
        DOSE_WEIGHT_SCALE = 9.0
        dose_weight = 1.0 + DOSE_WEIGHT_SCALE * true_dose.clamp(max=0.96)
        loss_mse = ((pred_dose - true_dose) ** 2 * dose_weight).mean()

        # ------ 2. L_V-Type (Dual-Tier DVH) ------
        loss_optional = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        loss_mandatory = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)

        organ_map = {"Bladder": bladder_mask, "Anorectum": rectum_mask, "Bag_Bowel": bag_bowel_mask}
        
        for organ_name, mask in organ_map.items():
            for rule in constraints_dict["v_type"].get(organ_name, []):
                v_frac = self.calculate_dvh_volume(pred_dose, mask, rule["norm_dose"])
                if not math.isnan(rule["optimal_v"]):
                    viol_opt = torch.relu(v_frac - rule["optimal_v"])
                    loss_optional = loss_optional + viol_opt ** 2
                viol_mand = torch.relu(v_frac - rule["mandatory_v"])
                loss_mandatory = loss_mandatory + viol_mand ** 2

        # ------ 3. L_PTV (D-Type coverage) ------
        ptv_channel = inputs[:, 1:2, ...]
        
        # Extract discrete PTV masks with prescription-aware targets
        sib_targets = {}
        for p_key, dose_val in SIB_ORDER:
            mask = (torch.isclose(ptv_channel, torch.tensor(dose_val, device=pred_dose.device))).float()
            if mask.sum() > 0:
                # Ceiling at 107% of prescription
                ceil_val = dose_val * 1.07
                sib_targets[p_key] = {"mask": mask, "rx": dose_val, "ceil": ceil_val}

        loss_ptv = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        loss_ptv_max = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)

        for name, data in sib_targets.items():
            mask = data["mask"]
            if mask.sum() > 0:
                rx_norm = data["rx"] / prescription_gy
                ceil_norm = data["ceil"] / prescription_gy
                
                underdose_penalty = (torch.relu(rx_norm - pred_dose) ** 2) * mask
                loss_ptv += underdose_penalty.sum() / mask.sum()
                
                overdose_penalty = (torch.relu(pred_dose - ceil_norm) ** 2) * mask
                loss_ptv_max += overdose_penalty.sum() / mask.sum()

        # ------ Anti-Collapse ------
        ptv_n = ptv_mask.sum()
        loss_anticollapse = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 0:
            ptv_underdose_frac = (torch.relu(0.50 - pred_dose) * ptv_mask).sum() / ptv_n
            loss_anticollapse = ptv_underdose_frac ** 2

        # ------ Global Hard Ceiling ------
        K_CEIL = 100
        # Use 107% of max PTV dose as ceiling
        max_ptv_dose = max([data["ceil"] for data in sib_targets.values()]) if sib_targets else 75.0
        global_ceil_norm = (max_ptv_dose * 1.07) / prescription_gy
        
        body_pred = pred_dose[inputs[:, 4:5, ...] > 0.5]
        loss_global_ceil = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if body_pred.numel() > 0:
            ceil_violations = torch.relu(body_pred - global_ceil_norm)
            if ceil_violations.max() > 0:
                K = min(K_CEIL, ceil_violations.numel())
                top_violations, _ = torch.topk(ceil_violations, K)
                loss_global_ceil = (top_violations ** 2).sum() / K_CEIL

        # ------ L_Ring ------
        RING_THRESH = 62.0 / prescription_gy
        ring_n = ring_mask.sum()
        if ring_n > 0:
            ring_overdose = torch.relu(pred_dose - RING_THRESH) * ring_mask
            loss_ring = (ring_overdose ** 2).sum() / ring_n
        else:
            loss_ring = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)

        # ------ Background Suppression ------
        oar_exclusion = ptv_mask + bladder_mask + rectum_mask + ring_mask + bowel_mask + femur_mask + bag_bowel_mask
        bg_mask = (inputs[:, 4:5, ...] > 0.5).float() - oar_exclusion
        bg_mask = torch.clamp(bg_mask, min=0.0)
        beam_mask_ch = (inputs[:, 6:7, ...] > 0.5).float()
        in_beam_bg_mask = torch.clamp(bg_mask * beam_mask_ch, min=0.0)
        out_beam_bg_mask = torch.clamp(bg_mask * (1.0 - beam_mask_ch), min=0.0)

        loss_bg = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        IN_BEAM_CEIL = 15.0 / prescription_gy
        in_beam_pred = pred_dose[in_beam_bg_mask.bool()]
        if in_beam_pred.numel() > 0:
            in_beam_violations = torch.relu(in_beam_pred - IN_BEAM_CEIL)
            loss_bg = loss_bg + (in_beam_violations ** 2).mean()

        OUT_BEAM_CEIL = 2.0 / prescription_gy
        out_beam_pred = pred_dose[out_beam_bg_mask.bool()]
        if out_beam_pred.numel() > 0:
            out_beam_violations = torch.relu(out_beam_pred - OUT_BEAM_CEIL)
            loss_bg = loss_bg + 5.0 * (out_beam_violations ** 2).mean()

        # ------ L_Body ------
        body_mask = (inputs[:, 4:5, ...] > 0.5).float()
        outside_body_mask = 1.0 - body_mask
        ghost_dose = pred_dose * outside_body_mask
        n_outside_body = outside_body_mask.sum().clamp(min=1.0)
        loss_body = ghost_dose.sum() / n_outside_body

        # ------ L_smooth ------
        gd = pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :]
        gh = pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :]
        gw = pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1]
        loss_smooth = (torch.mean(gd ** 2) + torch.mean(gh ** 2) + torch.mean(gw ** 2))

        # ------ L_Laplacian ------
        laplacian_d = (pred_dose[:, :, 2:, :, :] - 2 * pred_dose[:, :, 1:-1, :, :] + pred_dose[:, :, :-2, :, :])
        laplacian_h = (pred_dose[:, :, :, 2:, :] - 2 * pred_dose[:, :, :, 1:-1, :] + pred_dose[:, :, :, :-2, :])
        laplacian_w = (pred_dose[:, :, :, :, 2:] - 2 * pred_dose[:, :, :, :, 1:-1] + pred_dose[:, :, :, :, :-2])
        loss_laplacian = (torch.mean(laplacian_d ** 2) + torch.mean(laplacian_h ** 2) + torch.mean(laplacian_w ** 2))

        # ------ L_Bowel ------
        BOWEL_OPT_THRESH = 45.0 / prescription_gy
        BOWEL_OPT_LIMIT = 0.30
        BOWEL_MAND_THRESH = 50.0 / prescription_gy
        BOWEL_MAND_LIMIT = 0.50
        MANDATORY_SCALE = 5.0

        bowel_v45 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_OPT_THRESH)
        bowel_v50 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_MAND_THRESH)
        loss_bowel_opt = torch.relu(bowel_v45 - BOWEL_OPT_LIMIT) ** 2
        loss_bowel_mand = torch.relu(bowel_v50 - BOWEL_MAND_LIMIT) ** 2
        loss_bowel = loss_bowel_opt + MANDATORY_SCALE * loss_bowel_mand

        # ------ L_Femur ------
        FEMUR_MAX_NORM = 40.0 / prescription_gy
        femur_n = femur_mask.sum().clamp(min=1.0)
        loss_femur = ((torch.relu(pred_dose - FEMUR_MAX_NORM) ** 2) * femur_mask).sum() / femur_n

        # ------ L_Penile ------
        penile_mask_ch = inputs[:, 5:6, ...]
        loss_penile = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        for rule in constraints_dict["v_type"].get("PenileBulb", []):
            v_frac = self.calculate_dvh_volume(pred_dose, penile_mask_ch, rule["norm_dose"])
            viol = torch.relu(v_frac - rule["mandatory_v"])
            loss_penile = loss_penile + viol ** 2

        # ------ Total Loss ------
        total = (
            self.lambda_mse * loss_mse
            + self.lambda_optional * loss_optional
            + self.lambda_mandatory * loss_mandatory
            + self.lambda_ptv * loss_ptv
            + self.lambda_ptv_max * loss_ptv_max
            + self.lambda_global_ceil * loss_global_ceil
            + self.lambda_ring * loss_ring
            + self.lambda_smooth * loss_smooth
            + self.lambda_laplacian * loss_laplacian
            + self.lambda_anticollapse * loss_anticollapse
            + self.lambda_body * loss_body
            + self.lambda_bowel * loss_bowel
            + self.lambda_femur * loss_femur
            + self.lambda_penile * loss_penile
            + self.lambda_bg * loss_bg
        )

        return total, {
            "mse": loss_mse.item(),
            "v_opt": loss_optional.item(),
            "v_mand": loss_mandatory.item(),
            "ptv": loss_ptv.item(),
            "ptv_max": loss_ptv_max.item(),
            "global_ceil": loss_global_ceil.item(),
            "ring": loss_ring.item(),
            "smooth": loss_smooth.item(),
            "laplacian": loss_laplacian.item(),
            "anticollapse": loss_anticollapse.item(),
            "body": loss_body.item(),
            "bowel": loss_bowel.item(),
            "femur": loss_femur.item(),
            "penile": loss_penile.item(),
            "bg": loss_bg.item(),
        }


# ===================================================================
# Transforms
# ===================================================================

class CreateDiscretePTVMapd(MapTransform):
    """Painter's Algorithm for SIB - lowest dose first, highest last."""
    def __call__(self, data):
        d = dict(data)
        discrete_ptv = torch.zeros_like(d["ch_0"])
        for p_key, dose_val in SIB_ORDER:
            if p_key in d:
                mask = d[p_key] >= 0.5
                discrete_ptv = torch.where(
                    mask,
                    torch.tensor(dose_val, dtype=discrete_ptv.dtype, device=discrete_ptv.device),
                    discrete_ptv,
                )
                del d[p_key]
        d["discrete_ptv"] = discrete_ptv
        return d


class Create5ClassCropMaskd(MapTransform):
    """Create crop mask with 5 classes for RandCropByLabelClassesd."""
    def __init__(self, keys, body_thresh_hu=-300.0):
        super().__init__(keys)
        self.body_thresh = body_thresh_hu
    
    def __call__(self, data):
        d = dict(data)
        # Class 0: Outside body
        # Class 1: PTV core (discrete_ptv >= 55 Gy)
        # Class 2: PTV shell (35 <= discrete_ptv < 55)
        # Class 3: PTV rim (discrete_ptv < 35)
        # Class 4: Body non-PTV
        
        body = d["ch_0"] > self.body_thresh
        ptv = d["discrete_ptv"] > 0
        
        high_ptv = d["discrete_ptv"] >= 55
        med_ptv = (d["discrete_ptv"] >= 35) & (d["discrete_ptv"] < 55)
        low_ptv = (d["discrete_ptv"] > 0) & (d["discrete_ptv"] < 35)
        
        crop_mask = torch.zeros_like(d["ch_0"], dtype=torch.long)
        crop_mask[0] = (~body).long() * 0  # Outside
        crop_mask[0] = crop_mask[0] + (body & ~ptv).long() * 4  # Body non-PTV
        crop_mask[0] = crop_mask[0] + low_ptv.long() * 3  # PTV rim
        crop_mask[0] = crop_mask[0] + med_ptv.long() * 2  # PTV shell
        crop_mask[0] = crop_mask[0] + high_ptv.long() * 1  # PTV core
        
        d["crop_mask"] = crop_mask[0]
        return d


class CreateFalloffRingd(MapTransform):
    """Create 5mm ring around PTV for falloff penalty."""
    def __call__(self, data):
        d = dict(data)
        ptv = d["ch_1"] > 0.5  # Binary PTV mask
        
        # Dilate by 5mm
        dilated = F.max_pool3d(
            ptv.float().unsqueeze(0),
            kernel_size=(5, 5, 5),
            stride=1,
            padding=2
        )[0]
        
        ring = (dilated > 0) & (~ptv)
        d["ring_mask"] = ring.float()
        return d


# ===================================================================
# Data Loading with Prescription Metadata
# ===================================================================

def get_data_dicts_with_prescriptions():
    """Load data dictionaries with prescription metadata."""
    metadata = load_patient_metadata()
    
    data_dicts = []
    for patient_id, prescription_scheme in metadata.items():
        # Check if all required files exist
        base_path = os.path.join(IMAGES_DIR, patient_id)
        required_files = [f"{base_path}_{ch}.nii.gz" for ch in CHANNELS]
        label_file = os.path.join(LABELS_DIR, f"{patient_id}.nii.gz")
        
        if not all(os.path.exists(f) for f in required_files + [label_file]):
            continue
        
        patient_dict = {
            f"ch_{i}": f"{base_path}_{ch}.nii.gz"
            for i, ch in enumerate(CHANNELS)
        }
        patient_dict["dose_label"] = label_file
        
        # Add individual PTV files if they exist
        for ptv_key in SIB_KEYS:
            ptv_file = f"{base_path}_{ptv_key}.nii.gz"
            if os.path.exists(ptv_file):
                patient_dict[ptv_key] = ptv_file
        
        # Add prescription metadata
        patient_dict["prescription_scheme"] = prescription_scheme
        patient_dict["prescription_gy"] = PRESCRIPTION_SCHEMES.get(prescription_scheme, 75.0)
        
        data_dicts.append(patient_dict)
    
    return data_dicts


# ===================================================================
# Training Setup
# ===================================================================

def setup_training():
    """Initialize model, optimizer, and data loaders."""
    # Assign prescriptions if needed
    assign_prescriptions_to_patients()
    
    # Get data
    data_dicts = get_data_dicts_with_prescriptions()
    n_total = len(data_dicts)
    print(f"Found {n_total} patients with prescription metadata")
    
    # Split by prescription scheme to ensure balanced validation
    scheme_groups = {}
    for d in data_dicts:
        scheme = d["prescription_scheme"]
        if scheme not in scheme_groups:
            scheme_groups[scheme] = []
        scheme_groups[scheme].append(d)
    
    # Stratified split - preserve proportion of each scheme
    train_files = []
    val_files = []
    
    val_frac = float(os.environ.get("VAL_SPLIT", "0.20"))
    for scheme, files in scheme_groups.items():
        n_val = max(1, round(len(files) * val_frac))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])
    
    print(f"Split: {len(train_files)} train / {len(val_files)} val")
    for scheme in scheme_groups:
        n_train = sum(1 for d in train_files if d["prescription_scheme"] == scheme)
        n_val = sum(1 for d in val_files if d["prescription_scheme"] == scheme)
        print(f"  {scheme}: {n_train} train / {n_val} val")
    
    return train_files, val_files


# ===================================================================
# Main Training Loop
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Generalized Prescription Training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Setup
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/train_generalized_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    train_files, val_files = setup_training()
    
    if len(train_files) == 0:
        print("ERROR: No training files found. Run assign_prescriptions_to_patients() first.")
        return
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 8 channels: 7 original + 1 prescription channel
    model = UNet(
        spatial_dims=3,
        in_channels=8,  # 7 base + 1 prescription
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # Loss with curriculum ramp
    loss_fn = GeneralizedPhysicsDoseLoss(
        lambda_mse=25.0,
        lambda_global_ceil=2.0,  # Active from epoch 0
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    
    # Training loop placeholder - full implementation would include:
    # - Transform pipeline with prescription channel creation
    # - Per-batch constraint loading
    # - Validation across all prescription schemes
    # - Model checkpointing
    
    print("\\n" + "="*60)
    print("Generalized Prescription Training Setup Complete")
    print("="*60)
    print(f"Model: UNet with 8 input channels (7 base + 1 prescription)")
    print(f"Prescription schemes: {list(PRESCRIPTION_SCHEMES.keys())}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")
    print(f"\\nTo complete training, implement:")
    print("  1. Transform pipeline with prescription channel")
    print("  2. Per-batch constraint loading")
    print("  3. Training loop with prescription-aware loss")
    print("="*60)


if __name__ == "__main__":
    main()

