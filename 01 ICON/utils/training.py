"""
train_dummy_physics.py
======================
Physics-Guided Neural Network (PGNN) training script for prostate
radiotherapy dose prediction.  Implements:
  - CSV-driven clinical constraint parsing (N0 class)
  - Differentiable DVH via steep sigmoid (V-Type constraints)
  - Dual-Tier hinge penalties (Optimal / Mandatory)
  - Absolute dose penalties (D-Type: PTV coverage + max dose)
  - Spatial smoothness regularisation
  - CosineAnnealingLR scheduler
  - Mixed-precision (torch.cuda.amp) safe throughout
"""

import os
import glob
import math
import csv
import logging
from datetime import datetime

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
import monai
# pyrefly: ignore [missing-import]
from monai.data import PersistentDataset, DataLoader, list_data_collate
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    RandFlipd,
    ToTensord,
    ConcatItemsd,
    MapTransform,
    RandCropByLabelClassesd,
    DeleteItemsd,
)
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_SCRIPT_DIR, "..", "config", "config.yml")

with open(_CONFIG_FILE, "r") as _f:
    config = yaml.safe_load(_f)

# ── Channel-layout helpers (derived from config once at import time) ────────
# Index of every channel key
_CH_INDEX = {ch["key"]: ch["index"] for ch in config["channels"]}
# Spacingd interpolation modes in channel-index order
_CH_MODES_TUPLE = tuple(
    ch["interpolation"]
    for ch in sorted(config["channels"], key=lambda c: c["index"])
)
# Special-role indices
_BODY_CH_IDX   = next(ch["index"] for ch in config["channels"] if ch.get("role") == "Body_Mask")
_PTV_CH_IDX    = next(ch["index"] for ch in config["channels"] if ch.get("role") == "PTV_binary")
_BEV_CH_IDX    = next(ch["index"] for ch in config["channels"] if ch.get("role") == "BEV_Beam")
# SDM channels keyed by canonical organ name
_SDM_ORGAN_IDX = {
    ch["organ"]: ch["index"]
    for ch in config["channels"]
    if ch.get("role") == "SDM"
}
# Binary-mask channels keyed by canonical organ name
_BIN_ORGAN_IDX = {
    ch["organ"]: ch["index"]
    for ch in config["channels"]
    if ch.get("role") == "Binary_Mask"
}
_PENILE_CH_IDX = _BIN_ORGAN_IDX.get("Penile_Bulb", 5)

# ── OAR helpers ─────────────────────────────────────────────────────────────
_OAR_BY_CANONICAL  = {oar["canonical"]: oar for oar in config["organs_at_risk"]}
# Extra-file mask keys (bowel_mask, femur_mask …)
_EXTRA_MASK_OARS   = [
    oar for oar in config["organs_at_risk"] if "extra_file_key" in oar
]
_EXTRA_MASK_KEYS   = [oar["extra_file_key"] for oar in _EXTRA_MASK_OARS]
# SDM OARs that feed into organ_map in PhysicsGuidedDoseLoss.forward()
_SDM_LOSS_OARS = [
    oar for oar in config["organs_at_risk"]
    if oar.get("requires_sdm") and "sdm_channel_key" in oar
]
# Crop-class config
_BODY_HU_THRESHOLD = float(config["body_hu_threshold"])
_CROP_CLASSES = config["crop_classes"]          # list of dicts
_CROP_RATIOS  = [cc["sample_ratio"] for cc in _CROP_CLASSES]

# ===================================================================

# ===================================================================
DATA_DIR = os.environ.get("DATA_DIR", config["dataset"]["data_dir"])
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

CHANNELS = [f"{ch['index']:04d}" for ch in sorted(config["channels"], key=lambda c: c["index"])]
TARGET_SPACING = tuple(config["dataset"]["target_spacing"])
PATCH_SIZE = tuple(config["dataset"]["patch_size"])

PRESCRIPTION_DOSE_GY = config["clinical_targets"]["prescription_dose_gy"]
CONSTRAINT_CSV = os.environ.get("CONSTRAINT_CSV", config["dataset"]["constraint_csv"])

GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", config["training"]["grad_accum_steps"]))

VAL_EVERY_N_EPOCHS = int(os.environ.get("VAL_EVERY_N_EPOCHS", config["training"]["val_every_n_epochs"]))

WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", config["training"]["warmup_epochs"]))

# PTV target rx values derived from config (no hardcoded names/doses)
_TARGETS_SORTED   = sorted(config["clinical_targets"]["targets"], key=lambda t: t["rx_gy"], reverse=True)
_PTV_PRIMARY_RX   = float(_TARGETS_SORTED[0]["rx_gy"])
_PTV_SECONDARY_RX = float(_TARGETS_SORTED[1]["rx_gy"]) if len(_TARGETS_SORTED) > 1 else _PTV_PRIMARY_RX
_PTV_PRIMARY_NAME   = _TARGETS_SORTED[0]["name"]
_PTV_SECONDARY_NAME = _TARGETS_SORTED[1]["name"] if len(_TARGETS_SORTED) > 1 else _TARGETS_SORTED[0]["name"]

# ===================================================================

# ===================================================================

def load_clinical_constraints(csv_path=None, patient_class="N0"):
    """
    Load clinical constraints from config.yml inline block.
    Falls back to parsing csv_path if config inline block is absent.
    Returns the same structure as before:
      {"v_type": {organ: [...]}, "d_type": {"PTV_max_dose_gy": float, "PTV_coverage": [...]}}
    """
    rx = PRESCRIPTION_DOSE_GY

    # ---- Try inline YAML first -----------------------------------------
    inline = config.get("constraints", {}).get("v_type")
    if inline:
        v_constraints = {}
        for organ_name, rows in inline.items():
            entries = []
            for row in rows:
                if math.isnan(float(row.get("mandatory_v", float("nan")))):
                    continue
                entries.append({
                    "dose_gy":     float(row["dose_gy"]),
                    "norm_dose":   float(row["dose_gy"]) / rx,
                    "optimal_v":   float(row.get("optimal_v", float("nan"))),
                    "mandatory_v": float(row["mandatory_v"]),
                })
        v_constraints[organ_name] = sorted(entries, key=lambda r: r["dose_gy"])
        # PTV constraints not in inline block — return defaults
        return {
            "v_type": v_constraints,
            "d_type": {"PTV_max_dose_gy": None, "PTV_coverage": []},
        }

    # ---- CSV fallback --------------------------------------------------
    if csv_path is None:
        csv_path = CONSTRAINT_CSV
    v_accum = {}
    ptv_coverage = []
    ptv_max_dose_gy = None
    nplus_suffix = "_Nplus"
    # Collect canonical V-type organ names from OAR config
    v_type_organs = {
        oar["csv_constraint_name"]: oar["canonical"]
        for oar in config["organs_at_risk"]
        if "csv_constraint_name" in oar
    }
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name        = row["Name"].strip()
            struct_type = row["Type"].strip()
            ctype       = row["Constraint_Type"].strip()
            c_val_raw   = row["Constraint_Value"].strip()
            c_unit      = row["Constraint_Unit"].strip()
            obj_val_raw = row["Objective_Value"].strip()
            obj_unit    = row["Objective_Unit"].strip()
            obj_type    = row["Objective_Type"].strip()
            if not obj_val_raw:
                continue
            is_nplus = name.endswith(nplus_suffix)
            if patient_class == "N0" and is_nplus:
                continue
            if patient_class == "N+" and not is_nplus:
                continue
            canonical = name[: -len(nplus_suffix)] if is_nplus else name
            if ctype == "V" and canonical in v_type_organs:
                if obj_unit.strip() != "%":
                    continue
                dose_thresh_gy = float(c_val_raw)
                obj_value      = float(obj_val_raw)
                key = (canonical, dose_thresh_gy)
                if key not in v_accum:
                    v_accum[key] = {"optimal_v": float("nan"), "mandatory_v": float("nan")}
                if obj_type == "Optimal":
                    v_accum[key]["optimal_v"] = obj_value
                elif obj_type == "Mandatory":
                    v_accum[key]["mandatory_v"] = obj_value
            if (ctype == "D" and struct_type == "PTV" and c_val_raw == "Max" and c_unit == "Gy"):
                if obj_type == "Mandatory":
                    ptv_max_dose_gy = float(obj_val_raw)
            if (ctype == "D" and struct_type == "PTV" and c_unit == "%" and obj_unit == "%"):
                try:
                    percentile = float(c_val_raw)
                except ValueError:
                    continue
                if percentile >= 90 and obj_type == "Mandatory":
                    ptv_coverage.append({"metric": f"D{int(percentile)}", "fraction": float(obj_val_raw)})
    v_constraints = {oar["canonical"]: [] for oar in config["organs_at_risk"] if "csv_constraint_name" in oar}
    for (organ, dose_gy), tiers in sorted(v_accum.items(), key=lambda x: x[0][1]):
        if math.isnan(tiers["mandatory_v"]):
            continue
        v_constraints.setdefault(organ, []).append({
            "dose_gy":     dose_gy,
            "norm_dose":   dose_gy / rx,
            "optimal_v":   tiers["optimal_v"],
            "mandatory_v": tiers["mandatory_v"],
        })
    return {
        "v_type": v_constraints,
        "d_type": {"PTV_max_dose_gy": ptv_max_dose_gy, "PTV_coverage": ptv_coverage},
    }


# ===================================================================

# ===================================================================

class PhysicsGuidedDoseLoss(nn.Module):
    """
    Implements  L_total = λ_mse · L_MSE
                        + λ_opt · L_V-opt
                        + λ_mand · L_V-mand
                        + λ_ptv · L_PTV
                        + λ_smooth · L_smooth

    All operations are differentiable tensor ops.  No Python if/else
    on voxel values — masks are multiplied, not indexed with booleans
    where gradients are needed.
    """

    def __init__(
        self,
        constraints_dict,
        lambda_mse=10.0,            
        lambda_optimal=2.0,         
        lambda_mandatory=50.0,      
        lambda_ptv=15.0,           
        lambda_smooth=1.0,          
        lambda_ring=15.0,           
        lambda_anticollapse=50.0,   
        lambda_ptv_max=150.0,       
        lambda_homogeneity=30.0,    
        lambda_laplacian=5.0,       
        lambda_bowel=15.0,          
        lambda_femur=10.0,          
        lambda_global_ceil=2.0,     
        lambda_shell_inner=0.0,     
        lambda_shell_outer=0.0,     
        lambda_penile=10.0,         # Penile Bulb V47Gy ≤ 50% (v3)
        lambda_bg=15.0,
        k_steepness=50.0,           
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.constraints = constraints_dict
        self.lambda_mse = lambda_mse
        self.lambda_optimal = lambda_optimal
        self.lambda_mandatory = lambda_mandatory
        self.lambda_ptv = lambda_ptv
        self.lambda_smooth = lambda_smooth
        self.lambda_ring = lambda_ring
        self.lambda_anticollapse = lambda_anticollapse
        self.lambda_ptv_max = lambda_ptv_max
        self.lambda_homogeneity = lambda_homogeneity
        self.lambda_laplacian = lambda_laplacian
        self.lambda_bowel = lambda_bowel
        self.lambda_femur = lambda_femur
        self.lambda_global_ceil = lambda_global_ceil
        self.lambda_shell_inner = lambda_shell_inner
        self.lambda_shell_outer = lambda_shell_outer
        self.lambda_body = 20.0    
        self.lambda_penile = lambda_penile   # Penile Bulb (v3)
        self.lambda_bg = lambda_bg
        self.k = k_steepness

    # --- Differentiable DVH volume fraction --------------------------
    def calculate_dvh_volume(self, predicted_dose, organ_mask, norm_dose_threshold):
        """
        V^pred_{D_ref} = (1/N_OAR) * Σ_{i∈OAR} σ(k·(D_i - D_ref))
        Optimized: Uses boolean indexing to avoid computing sigmoids on background voxels.
        """
        # Extract only the voxels physically inside the organ
        organ_voxels = predicted_dose[organ_mask.bool()]
        n_organ = organ_voxels.numel()

        if n_organ == 0:
            return torch.tensor(0.0, device=predicted_dose.device, dtype=predicted_dose.dtype)

        # Compute steep sigmoid exclusively on the isolated organ voxels
        step_approx = torch.sigmoid(self.k * (organ_voxels - norm_dose_threshold))

        volume_fraction = step_approx.sum() / n_organ
        return volume_fraction

    # --- Forward pass ------------------------------------------------
    def forward(self, pred_dose, true_dose, bladder_mask, rectum_mask,
                ptv_mask, ring_mask, inputs, bowel_mask, femur_mask):
        """
        Parameters
        ----------
        pred_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        true_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        bladder_mask: (B, 1, D, H, W)  — binary float
        rectum_mask : (B, 1, D, H, W)  — binary float
        ptv_mask    : (B, 1, D, H, W)  — binary float
        ring_mask   : (B, 1, D, H, W)  — binary float, 5mm shell around PTV
        inputs      : (B, 6, D, H, W)  — full input tensor
        bowel_mask  : (B, 1, D, H, W)  — Bag_Bowel binary float (may be all zeros)
        femur_mask  : (B, 1, D, H, W)  — merged Femur_Head_L+R binary float
        """
        # ------ 1. L_MSE (dose-weighted) --------------------------------
        
        DOSE_WEIGHT_SCALE = 9.0
        dose_weight = 1.0 + DOSE_WEIGHT_SCALE * true_dose.clamp(max=0.96)
        loss_mse = ((pred_dose - true_dose) ** 2 * dose_weight).mean()

        # ------ 2. L_V-Type (Dual-Tier DVH) -------------------------
        loss_optional = torch.tensor(0.0, device=pred_dose.device,
                                    dtype=pred_dose.dtype)
        loss_mandatory = torch.tensor(0.0, device=pred_dose.device,
                                      dtype=pred_dose.dtype)

        organ_map = {
            oar["canonical"]: (inputs[:, _SDM_ORGAN_IDX[oar["canonical"]]:_SDM_ORGAN_IDX[oar["canonical"]]+1, ...] <= 0.0).float()
            for oar in _SDM_LOSS_OARS
        }
        # Penile Bulb is handled separately below (has its own lambda).

        for organ_name, mask in organ_map.items():
            for rule in self.constraints["v_type"].get(organ_name, []):
                v_frac = self.calculate_dvh_volume(
                    pred_dose, mask, rule["norm_dose"]
                )
                
                if not math.isnan(rule["optimal_v"]):
                    viol_opt = torch.relu(v_frac - rule["optimal_v"])
                    loss_optional = loss_optional + viol_opt ** 2

                viol_mand = torch.relu(v_frac - rule["mandatory_v"])
                loss_mandatory = loss_mandatory + viol_mand ** 2

        # ------ 3. L_PTV  (D-Type coverage) -------------------------
        # PTV channel index from config
        ptv_channel = inputs[:, _PTV_CH_IDX:_PTV_CH_IDX+1, ...]
        
        # Dynamically extract independent masks — values match Gy magnitudes from CreateDiscretePTVMapd
        ptv60_mask = (torch.isclose(ptv_channel, torch.tensor(60.0, device=pred_dose.device))).float()
        ptv44_mask = (torch.isclose(ptv_channel, torch.tensor(44.0, device=pred_dose.device))).float()
        ptv55_mask = (torch.isclose(ptv_channel, torch.tensor(55.0, device=pred_dose.device))).float()
        ptv36_mask = (torch.isclose(ptv_channel, torch.tensor(36.0, device=pred_dose.device))).float()
        ptv25_mask = (torch.isclose(ptv_channel, torch.tensor(25.0, device=pred_dose.device))).float()
        ptv54_mask = (torch.isclose(ptv_channel, torch.tensor(54.0, device=pred_dose.device))).float()

        # Map targets to their discrete extraction masks with explicit clinical floors/ceilings
        sib_targets = {
            lvl["name"].lower(): {
                "mask": (torch.isclose(
                    ptv_channel,
                    torch.tensor(lvl["rx_gy"], device=pred_dose.device)
                )).float(),
                "rx":   lvl["rx_gy"],
                "ceil": lvl["ceil_gy"],
            }
            for lvl in config["clinical_targets"]["targets"]
        }

        loss_ptv = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        loss_ptv_max = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        loss_homogeneity = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype) # Kept at 0.0 to avoid breaking return dict

        for name, data in sib_targets.items():
            mask = data["mask"]
            if mask.sum() > 0:
                rx_norm = data["rx"] / PRESCRIPTION_DOSE_GY
                ceil_norm = data["ceil"] / PRESCRIPTION_DOSE_GY
                
                # The Floor: Strict penalty for any underdosing below Rx
                underdose_penalty = (torch.relu(rx_norm - pred_dose) ** 2) * mask
                loss_ptv += underdose_penalty.sum() / mask.sum()

                # The Ceiling: Strict penalty for any overdosing above ICRU limit
                overdose_penalty = (torch.relu(pred_dose - ceil_norm) ** 2) * mask
                loss_ptv_max += overdose_penalty.sum() / mask.sum()
                
                # Between rx_norm and ceil_norm, the gradient is exactly 0.

        # ------ 3a. Homogeneity Penalty -----------------------------
        # (Computed inside the SIB loop above)

        # ------ 3b. Anti-Collapse Safety Net ------------------------
        ptv_n = ptv_mask.sum()
        loss_anticollapse = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 0:
            ptv_underdose_frac = (torch.relu(0.50 - pred_dose) * ptv_mask).sum() / ptv_n
            loss_anticollapse = ptv_underdose_frac ** 2

        # ------ Global Hard Ceiling (Fixed-K=1000 Top-K MSE) ------
        # 64.2 Gy = 107% of PTV60 — hard clinical ceiling
        # K=1000 ≈ 4 cc of tissue at 1.27×1.27×2.5 mm spacing.
        # Dividing by constant K_CEIL (not by n_violations) avoids both
        # the volume-dilution trap (.mean() over 2M voxels) and the
        # denominator-inflation trap (dividing by N_violating).
        K_CEIL = 100
        global_ceil_norm = config["physics_engine"]["thresholds"]["global_ceil_gy"] / PRESCRIPTION_DOSE_GY
        body_pred = pred_dose[inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5]

        loss_global_ceil = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if body_pred.numel() > 0:
            ceil_violations = torch.relu(body_pred - global_ceil_norm)
            if ceil_violations.max() > 0:
                K = min(K_CEIL, ceil_violations.numel())
                top_violations, _ = torch.topk(ceil_violations, K)
                loss_global_ceil = (top_violations ** 2).sum() / K_CEIL

        # ------ 5. L_Ring (Falloff shell penalty) -------------------
        
        RING_THRESH = config["physics_engine"]["thresholds"]["ring_thresh_gy"] / PRESCRIPTION_DOSE_GY
        ring_n = ring_mask.sum()
        if ring_n > 0:
            ring_overdose = torch.relu(pred_dose - RING_THRESH) * ring_mask
            loss_ring = (ring_overdose ** 2).sum() / ring_n
        else:
            loss_ring = torch.tensor(0.0, device=pred_dose.device,
                                     dtype=pred_dose.dtype)

        # ------ 5b. Healthy Tissue Bath Suppression (Two-Tier BEV) -------------------
        # Isolate healthy tissue: Body - (PTV + OARs + Ring)
        oar_exclusion = ptv_mask + bladder_mask + rectum_mask + ring_mask + bowel_mask + femur_mask
        bg_mask = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float() - oar_exclusion
        bg_mask = torch.clamp(bg_mask, min=0.0)
        
        # Extract the beam corridors from config-driven BEV channel index
        beam_mask_ch = (inputs[:, _BEV_CH_IDX:_BEV_CH_IDX+1, ...] > 0.5).float()

        # Split the background into In-Beam and Out-of-Beam
        in_beam_bg_mask  = torch.clamp(bg_mask * beam_mask_ch, min=0.0)
        out_beam_bg_mask = torch.clamp(bg_mask * (1.0 - beam_mask_ch), min=0.0)

        loss_bg = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        
        # Tier 1: In-Beam Corridor Falloff (Relaxed to 15.0 Gy to match physical d_max)
        IN_BEAM_CEIL = config["physics_engine"]["thresholds"]["in_beam_ceil_gy"] / PRESCRIPTION_DOSE_GY
        in_beam_pred = pred_dose[in_beam_bg_mask.bool()]
        if in_beam_pred.numel() > 0:
            in_beam_violations = torch.relu(in_beam_pred - IN_BEAM_CEIL)
            loss_bg = loss_bg + (in_beam_violations ** 2).mean()

        # Tier 2: Out-of-Beam Absolute Dark Zone (physically impossible scatter)
        OUT_BEAM_CEIL = config["physics_engine"]["thresholds"]["out_beam_ceil_gy"] / PRESCRIPTION_DOSE_GY
        out_beam_pred = pred_dose[out_beam_bg_mask.bool()]
        if out_beam_pred.numel() > 0:
            out_beam_violations = torch.relu(out_beam_pred - OUT_BEAM_CEIL)
            _out_mult = config["physics_engine"]["thresholds"]["out_beam_multiplier"]
            loss_bg = loss_bg + _out_mult * (out_beam_violations ** 2).mean()

        loss_shell_inner = torch.tensor(0.0, device=pred_dose.device, 
                                        dtype=pred_dose.dtype)
        loss_shell_outer = torch.tensor(0.0, device=pred_dose.device, 
                                        dtype=pred_dose.dtype)

#        # ------ 5e. Concentric Shell Falloff Penalties ---------------
#
#        ptv_for_dilation = ptv_mask.float()  
#
#        dil_20mm = F.max_pool3d(
#            ptv_for_dilation,
#            kernel_size=(17, 33, 33),
#            stride=1,
#            padding=(8, 16, 16),
#        )
#        dil_40mm = F.max_pool3d(
#            ptv_for_dilation,
#            kernel_size=(33, 63, 63),
#            stride=1,
#            padding=(16, 31, 31),
#        )
#
#        shell_inner_mask = torch.clamp(dil_20mm - ptv_for_dilation, 0.0, 1.0)
#        shell_outer_mask = torch.clamp(dil_40mm - dil_20mm, 0.0, 1.0)
#
#        oar_exclusion = torch.clamp(
#            bladder_mask + rectum_mask, 0.0, 1.0
#        )
#        shell_inner_mask = shell_inner_mask * (1.0 - oar_exclusion)
#        shell_outer_mask = shell_outer_mask * (1.0 - oar_exclusion)
#
#        SHELL_INNER_CEIL = 45.0 / PRESCRIPTION_DOSE_GY  
#        loss_shell_inner = torch.tensor(0.0, device=pred_dose.device,
#                                        dtype=pred_dose.dtype)
#        inner_pred = pred_dose[shell_inner_mask.bool()]
#        if inner_pred.numel() > 0:
#            inner_violations = torch.relu(inner_pred - SHELL_INNER_CEIL)
#            if inner_violations.max() > 0:
#                K_inner = max(int(0.001 * inner_pred.numel()), 10)
#                K_inner = min(K_inner, inner_pred.numel())
#                topk_inner, _ = torch.topk(inner_violations, K_inner)
#                loss_shell_inner = topk_inner.sum()
#
#        SHELL_OUTER_CEIL = 30.0 / PRESCRIPTION_DOSE_GY  
#        loss_shell_outer = torch.tensor(0.0, device=pred_dose.device,
#                                        dtype=pred_dose.dtype)
#        outer_pred = pred_dose[shell_outer_mask.bool()]
#        if outer_pred.numel() > 0:
#            outer_violations = torch.relu(outer_pred - SHELL_OUTER_CEIL)
#            if outer_violations.max() > 0:
#                K_outer = max(int(0.001 * outer_pred.numel()), 10)
#                K_outer = min(K_outer, outer_pred.numel())
#                topk_outer, _ = torch.topk(outer_violations, K_outer)
#                loss_shell_outer = topk_outer.sum()

        # ------ 5c. L_Body (Anti-Ghost Suppression) ----------------------
        
        body_mask = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float()
        outside_body_mask = 1.0 - body_mask
        ghost_dose = pred_dose * outside_body_mask
        n_outside_body = outside_body_mask.sum().clamp(min=1.0)
        loss_body = ghost_dose.sum() / n_outside_body

        # ------ 6. L_smooth (Total Variation) -----------------------
        
        gd = pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :]
        gh = pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :]
        gw = pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1]
        loss_smooth = (torch.mean(gd ** 2) + torch.mean(gh ** 2)
                       + torch.mean(gw ** 2))

        # ------ 6b. L_Laplacian (Penumbra Continuity) ----------------
        
        laplacian_d = (pred_dose[:, :, 2:, :, :]
                       - 2 * pred_dose[:, :, 1:-1, :, :]
                       + pred_dose[:, :, :-2, :, :])
        laplacian_h = (pred_dose[:, :, :, 2:, :]
                       - 2 * pred_dose[:, :, :, 1:-1, :]
                       + pred_dose[:, :, :, :-2, :])
        laplacian_w = (pred_dose[:, :, :, :, 2:]
                       - 2 * pred_dose[:, :, :, :, 1:-1]
                       + pred_dose[:, :, :, :, :-2])
        loss_laplacian = (torch.mean(laplacian_d ** 2)
                          + torch.mean(laplacian_h ** 2)
                          + torch.mean(laplacian_w ** 2))

        # ------ 7. L_Bowel (Bag_Bowel — dual-tier) -------------------
        
        _thr = config["physics_engine"]["thresholds"]
        BOWEL_OPT_THRESH  = _thr["bowel_opt_gy"]  / PRESCRIPTION_DOSE_GY
        BOWEL_OPT_LIMIT   = _thr["bowel_opt_limit_frac"]
        BOWEL_MAND_THRESH = _thr["bowel_mand_gy"] / PRESCRIPTION_DOSE_GY
        BOWEL_MAND_LIMIT  = _thr["bowel_mand_limit_frac"]
        MANDATORY_SCALE   = 5.0

        bowel_v45 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_OPT_THRESH)
        bowel_v50 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_MAND_THRESH)
        loss_bowel_opt  = torch.relu(bowel_v45 - BOWEL_OPT_LIMIT)  ** 2
        loss_bowel_mand = torch.relu(bowel_v50 - BOWEL_MAND_LIMIT) ** 2
        loss_bowel = loss_bowel_opt + MANDATORY_SCALE * loss_bowel_mand

        # ------ 8. L_Femur (merged Femur_Head_L+R — D_max < 40 Gy) -------
        # v3 constraint: D_max < 40 Gy.  Penalise all voxels above 40/75 Gy.
        FEMUR_MAX_NORM = config["physics_engine"]["thresholds"]["femur_max_gy"] / PRESCRIPTION_DOSE_GY
        femur_n = femur_mask.sum().clamp(min=1.0)
        loss_femur = ((torch.relu(pred_dose - FEMUR_MAX_NORM) ** 2) * femur_mask).sum() / femur_n

        # Penile Bulb: read from config-driven binary channel index
        penile_mask_ch = inputs[:, _PENILE_CH_IDX:_PENILE_CH_IDX+1, ...]
        loss_penile = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        _penile_key = _OAR_BY_CANONICAL.get("Penile_Bulb", {}).get("csv_constraint_name", "Penile_Bulb")
        for rule in self.constraints["v_type"].get(_penile_key, []):
            v_frac = self.calculate_dvh_volume(pred_dose, penile_mask_ch, rule["norm_dose"])
            viol = torch.relu(v_frac - rule["mandatory_v"])
            loss_penile = loss_penile + viol ** 2

        # ------ Total -----------------------------------------------
        total = (
            self.lambda_mse          * loss_mse
            + self.lambda_optional   * loss_optional
            + self.lambda_mandatory  * loss_mandatory
            + self.lambda_ptv        * loss_ptv
            + self.lambda_ptv_max    * loss_ptv_max
            + self.lambda_global_ceil * loss_global_ceil
            + self.lambda_ring       * loss_ring
            + self.lambda_smooth     * loss_smooth
            + self.lambda_laplacian  * loss_laplacian
            + self.lambda_anticollapse * loss_anticollapse
            + self.lambda_shell_inner * loss_shell_inner
            + self.lambda_shell_outer * loss_shell_outer
            + self.lambda_homogeneity * loss_homogeneity
            + self.lambda_body        * loss_body
            + self.lambda_bowel       * loss_bowel
            + self.lambda_femur       * loss_femur
            + self.lambda_penile      * loss_penile
            + self.lambda_bg          * loss_bg
        )
        return total, {
            "mse":        loss_mse.item(),
            "v_opt":      loss_optional.item(),
            "v_mand":     loss_mandatory.item(),
            "ptv":        loss_ptv.item(),
            "ptv_max":    loss_ptv_max.item(),
            "global_ceil": loss_global_ceil.item(),
            "ring":       loss_ring.item(),
            "smooth":     loss_smooth.item(),
            "laplacian":  loss_laplacian.item(),
            "anticollapse": loss_anticollapse.item(),
            "shell_inner":  loss_shell_inner.item(),
            "shell_outer":  loss_shell_outer.item(),
            "homogeneity":  loss_homogeneity.item(),
            "body":       loss_body.item(),
            "bowel":      loss_bowel.item(),
            "femur":      loss_femur.item(),
            "penile":     loss_penile.item(),
            "bg":         loss_bg.item(),
        }

# ===================================================================

# ===================================================================

def get_data_dicts():
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.nii.gz")))
    data_dicts = []
    for label_path in label_files:
        patient_id = os.path.basename(label_path).replace(".nii.gz", "")
        pt_dict = {"dose_label": label_path}
        for i, ch in enumerate(CHANNELS):
            pt_dict[f"ch_{i}"] = os.path.join(
                IMAGES_DIR, f"{patient_id}_{ch}.nii.gz"
            )
        
        pt_dict["bowel_mask"] = os.path.join(IMAGES_DIR, f"{patient_id}_{oar['file_suffix']}.nii.gz") if (oar := _OAR_BY_CANONICAL.get("Bag_Bowel")) and "file_suffix" in oar else os.path.join(IMAGES_DIR, f"{patient_id}_bowel.nii.gz")
        pt_dict["femur_mask"] = os.path.join(IMAGES_DIR, f"{patient_id}_{oar['file_suffix']}.nii.gz") if (oar := _OAR_BY_CANONICAL.get("Femur")) and "file_suffix" in oar else os.path.join(IMAGES_DIR, f"{patient_id}_femur.nii.gz")
        
        # Load individual PTV structures for SIB mapping
        import glob as glb
        ptv_files = glb.glob(os.path.join(IMAGES_DIR, f"{patient_id}_PTV*.nii.gz"))
        for p_file in ptv_files:
            key_name = os.path.basename(p_file).replace(".nii.gz", "").split("_", 2)[-1]
            pt_dict[key_name] = p_file
            
        data_dicts.append(pt_dict)
    return data_dicts

class CreateDiscretePTVMapd(MapTransform):
    """
    Creates a single discrete integer PTV Map channel to handle SIB targets.
    Assigns unique, discrete integer values to different PTV volumes.
    Higher-dose regions overwrite lower-dose regions via Painter's Algorithm.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        # Dynamically load from config, preserving order (lowest to highest dose)
        self.processing_order = [
            (level["name"], level["rx_gy"])
            for level in config["clinical_targets"]["targets"]
        ]

    def __call__(self, data):
        d = dict(data)
        
        # Initialize discrete map with zeros, same shape as ch_0
        ref_tensor = d['ch_0']
        discrete_ptv = torch.zeros_like(ref_tensor)
        
        # Painter's Algorithm: lowest dose first, highest dose last
        for p_key, integer_id in self.processing_order:
            if p_key in d:
                mask = d[p_key] >= 0.5
                discrete_ptv = torch.where(mask, torch.tensor(integer_id, device=discrete_ptv.device), discrete_ptv)
                
                # Delete the raw key to save RAM
                del d[p_key]

        d['discrete_ptv'] = discrete_ptv
        return d

class Create5ClassCropMaskd(MapTransform):
    """
    Builds a 5-class label map for RandCropByLabelClassesd.

    Must run BEFORE NormalizeIntensityd so ch_0 still holds raw HU values.

    Class 0 — Air / Absolute Background : CT < body_thresh_hu (outside body)
    Class 1 — Healthy Tissue            : inside body, not PTV/Bladder/Rectum
    Class 2 — PTV                       : PTV channel >= 0.5
    Class 3 — Bladder                   : Bladder SDM <= 0.0
    Class 4 — Rectum                    : Rectum SDM <= 0.0

    Precedence (last-write wins, highest priority last):
      Air → Healthy → PTV → Bladder → Rectum

    Sampling ratios in train_transforms: [0, 1, 1, 1, 1]
      → Class 0 (Air): 0 %   (never waste compute on empty space)
      → Classes 1-4  : 25 % each  (perfectly balanced with num_samples=4)
    """

    def __init__(self, keys, body_thresh_hu: float = _BODY_HU_THRESHOLD,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.body_thresh_hu = body_thresh_hu
        # Build crop-class rules from config
        self._sdm_classes = [
            (cc["sdm_channel_key"], cc["class_id"])
            for cc in _CROP_CLASSES
            if cc.get("source") == "sdm_channel"
        ]

    def __call__(self, data):
        d = dict(data)
        ct  = d["ch_0"]
        ptv = d["ch_1"] >= 0.5
        body = ct >= self.body_thresh_hu
        crop_mask = torch.zeros_like(ct, dtype=torch.float32)
        # Class 1: Healthy tissue inside body
        crop_mask[body] = 1.0
        # Class 2: PTV
        crop_mask[ptv] = 2.0
        # Classes from SDM channels (Bladder=3, Anorectum=4, …)
        for ch_key, cls_id in self._sdm_classes:
            crop_mask[d[ch_key] <= 0.0] = float(cls_id)
        d["crop_mask"] = crop_mask
        return d

# ===================================================================

# ===================================================================

def compute_ring_mask(ptv_binary: torch.Tensor) -> torch.Tensor:
    """
    3-D morphological dilation of a binary PTV mask, then subtract the
    original PTV to obtain a hollow 5 mm shell.

    Uses F.max_pool3d — pure PyTorch, works on any device (CPU / CUDA).

    Kernel (5, 9, 9) with padding (2, 4, 4):
      Z-axis : 2 voxels × 2.5 mm/vox = 5.0 mm
      XY-axes: 4 voxels × 1.27 mm/vox ≈ 5.1 mm

    Args:
        ptv_binary: float tensor, either (1,D,H,W) or (B,1,D,H,W).
    Returns:
        ring float tensor with same shape.
    """
    needs_batch = ptv_binary.ndim == 4          
    if needs_batch:
        ptv_binary = ptv_binary.unsqueeze(0)    

    dilated = F.max_pool3d(
        ptv_binary.float(),
        kernel_size=(5, 9, 9),
        stride=1,
        padding=(2, 4, 4),
    )
    ring = torch.clamp(dilated - ptv_binary.float(), min=0.0)

    if needs_batch:
        ring = ring.squeeze(0)                  
    return ring

class CreateFalloffRingd(MapTransform):
    """
    Generates the 5 mm falloff ring mask around the PTV.

    Must run AFTER Spacingd (so ch_1 is at TARGET_SPACING).
    Can run before or after NormalizeIntensityd (uses ch_1 only).
    Stores result under the key `ring_mask` for RandCropByLabelClassesd.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        ptv = (d["ch_1"] >= 0.5).float()   
        d["ring_mask"] = compute_ring_mask(ptv)
        return d

SIB_KEYS = [lvl["name"] for lvl in config["clinical_targets"]["targets"]]
ALL_KEYS = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6",
            "bowel_mask", "femur_mask",
            "dose_label"] + SIB_KEYS

# Keys that survive after CreateDiscretePTVMapd deletes the individual PTVs
CROPPED_KEYS = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6",
                "bowel_mask", "femur_mask", "dose_label", 
                "discrete_ptv", "ring_mask"]

# --- 1. Deterministic Transforms (Cached) ---
train_transforms_det = Compose(
    [
        LoadImaged(keys=ALL_KEYS, allow_missing_keys=True),
        EnsureChannelFirstd(keys=ALL_KEYS, allow_missing_keys=True),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest", "nearest",  # ch_0 to ch_6
                  "nearest", "nearest",   # bowel_mask, femur_mask
                  "bilinear") + ("nearest",)*len(SIB_KEYS),
            allow_missing_keys=True
        ),
        CreateDiscretePTVMapd(keys=["ch_0"]),
        
        Create5ClassCropMaskd(keys=["ch_0"]),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        
        CreateFalloffRingd(keys=["ch_1"]),
    ]
)

# --- 2. Random Transforms (On the fly) ---
train_transforms_rand = Compose(
    [
        RandCropByLabelClassesd(
            keys=CROPPED_KEYS,
            label_key="crop_mask",
            spatial_size=PATCH_SIZE,
            num_classes=len(_CROP_CLASSES),
            ratios=_CROP_RATIOS,
            num_samples=2,
        ),
        DeleteItemsd(keys=["crop_mask"]),
        ConcatItemsd(keys=["ch_0", "discrete_ptv", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"], name="image"),
        ToTensord(keys=["image", "dose_label", "ring_mask", "bowel_mask", "femur_mask"]),
    ]
)

# Validation transforms can be fully deterministic since they don't crop
val_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS, allow_missing_keys=True),
        EnsureChannelFirstd(keys=ALL_KEYS, allow_missing_keys=True),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest", "nearest",  # ch_0 – ch_6
                  "nearest", "nearest",   # bowel_mask, femur_mask
                  "bilinear") + ("nearest",)*len(SIB_KEYS),
            allow_missing_keys=True
        ),
        CreateDiscretePTVMapd(keys=["ch_0"]),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0", "discrete_ptv", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"], name="image"),
        ToTensord(keys=["image", "dose_label", "bowel_mask", "femur_mask"]),
    ]
)

# ===================================================================

# ===================================================================

def extract_binary_masks(inputs):
    """
    Convert the concatenated input tensor channels to binary masks.
    Channel layout (v4, 7 channels):
      0 = CT (normalized HU)
      1 = discrete PTV map (integer 1-6 per sub-volume)
      2 = Bladder SDM   (≤ 0 inside organ)
      3 = Anorectum SDM (≤ 0 inside organ)
      4 = Body Mask (binary)
      5 = Penile Bulb (binary)
      6 = BEV Beam Frustum (binary)

    Returns ptv_mask, bladder_mask, rectum_mask  — all (B,1,D,H,W) float.
    """
    ptv_mask = (inputs[:, _PTV_CH_IDX:_PTV_CH_IDX+1, ...] >= 0.5).float()
    # Build SDM masks from config
    sdm_masks = {
        canonical: (inputs[:, idx:idx+1, ...] <= 0.0).float()
        for canonical, idx in _SDM_ORGAN_IDX.items()
    }
    # Resolve SDM canonical names from config (Bladder, Anorectum, …)
    _bladder_key  = next((oar["canonical"] for oar in config["organs_at_risk"] if oar.get("sdm_channel_key") == "ch_2"), "Bladder")
    _anorect_key  = next((oar["canonical"] for oar in config["organs_at_risk"] if oar.get("sdm_channel_key") == "ch_3"), "Anorectum")
    bladder_mask = sdm_masks.get(_bladder_key, torch.zeros_like(ptv_mask))
    rectum_mask  = sdm_masks.get(_anorect_key, torch.zeros_like(ptv_mask))
    return ptv_mask, bladder_mask, rectum_mask

# ===================================================================

# ===================================================================

def main():
    # ---- Setup logging ---------------------------------------------
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_{timestamp}.log"
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training log started: {log_filename}")
    logger.info(f"PATCH_SIZE={PATCH_SIZE}, GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}")

    # ---- Load constraints ------------------------------------------
    logger.info(f"Loading clinical constraints (inline config) ...")
    print(f"Loading clinical constraints (inline config) ...")
    constraints = load_clinical_constraints()

    _v_type_oar_names = [oar["canonical"] for oar in config["organs_at_risk"] if "csv_constraint_name" in oar]
    for oar_name in _v_type_oar_names:
        n = len(constraints["v_type"].get(oar_name, []))
        print(f"  V-Type constraints: {oar_name}={n}")
    n_ptv = len(constraints["d_type"]["PTV_coverage"])
    ptv_max = constraints["d_type"]["PTV_max_dose_gy"]
    print(f"  D-Type constraints: PTV coverage rules={n_ptv}  PTV max={ptv_max} Gy")

    for oar_name in _v_type_oar_names:
        for r in constraints["v_type"].get(oar_name, []):
            print(f"    {oar_name}  V{r['dose_gy']}Gy  "
                  f"opt={r['optimal_v']:.2f}  mand={r['mandatory_v']:.2f}  "
                  f"norm_thresh={r['norm_dose']:.4f}")

    # ---- Dataset split (ratio-based, env-var overrideable) ----------
    print("\nFinding data...")
    data_dicts = get_data_dicts()
    n_total = len(data_dicts)
    print(f"Found {n_total} patients.")

    val_frac   = float(os.environ.get("VAL_SPLIT", "0.20"))
    n_val      = max(1, round(n_total * val_frac))
    n_train    = n_total - n_val
    train_files = data_dicts[:n_train]
    val_files   = data_dicts[n_train:]
    print(f"Split: {n_train} train  /  {n_val} val  (val_frac={val_frac:.0%})")

    # ---- Batch size and workers (env-var overrideable) ---------------
    
    batch_size   = int(os.environ.get("BATCH_SIZE", "1"))
    
    num_workers     = min(int(os.environ.get("NUM_WORKERS", "2")), 2)
    prefetch_factor = 1  
    print(f"Batch size={batch_size}  num_workers={num_workers}  prefetch={prefetch_factor}")

    cache_dir = os.path.join(DATA_DIR, "persistent_cache_physics")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n  Cache dir: {cache_dir}")
    print(f"  Patch size: {PATCH_SIZE}")
    print(f"  NOTE: If you see 'RuntimeError: PytorchStreamReader failed reading zip archive',")
    print(f"        clear the cache with: rm -rf {cache_dir}/*")

    # Wrap the PersistentDataset with standard Dataset to apply random transforms on the fly
    from monai.data import Dataset
    
    train_cache_ds = PersistentDataset(
        data=train_files, transform=train_transforms_det, cache_dir=cache_dir
    )
    train_ds = Dataset(data=train_cache_ds, transform=train_transforms_rand)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(int(os.environ.get("NUM_WORKERS", "2")), 2),
        prefetch_factor=1,  
        persistent_workers=False,  
        pin_memory=False,
        collate_fn=list_data_collate,
        drop_last=True,  
    )

    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,              
        shuffle=False,
        num_workers=min(int(os.environ.get("NUM_WORKERS", "2")), 2),  
        prefetch_factor=1,  
        persistent_workers=False,  
        pin_memory=False,
    )


    # ---- Model -----------------------------------------------------
    print("Building 3D U-Net...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  
        torch.backends.cuda.matmul.allow_tf32 = True  
        torch.backends.cudnn.allow_tf32 = True
        print(f"  CUDA: cuDNN benchmark=ON, TF32=ON")

    model = UNet(
        spatial_dims=3,
        in_channels=7,  # 7 inputs: CT, discrete_PTV, Bladder_SDM, Ano_SDM, Body, PenileBulb, BEV_Beam
        out_channels=1,
        channels=(16, 32, 64, 128),  
        strides=(2, 2, 2),           
        num_res_units=2,
    ).to(device)

    # ---- Loss / Optimizer / Scheduler ------------------------------
    
    loss_function = PhysicsGuidedDoseLoss(
        constraints_dict=constraints,
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
        lambda_shell_inner=0.0,     # disabled — not yet active
        lambda_shell_outer=0.0,     # disabled — not yet active
        lambda_bowel=0.0,           
        lambda_femur=0.0,           
        lambda_penile=0.0,          # Penile Bulb — ramp in
        lambda_bg=0.0,              # Added to prevent Epoch 0 shock
        k_steepness=50.0,
    )
    loss_function.lambda_body = 0.0  

    PHYSICS_TARGET_LAMBDAS = {
        "lambda_optional":     config["physics_engine"]["target_lambdas"]["optional"],
        "lambda_mandatory":    config["physics_engine"]["target_lambdas"]["mandatory"],
        "lambda_ptv":          config["physics_engine"]["target_lambdas"]["ptv"],
        "lambda_ptv_max":      config["physics_engine"]["target_lambdas"]["ptv_max"],
        "lambda_ring":         config["physics_engine"]["target_lambdas"]["ring"],
        "lambda_smooth":       config["physics_engine"]["target_lambdas"]["smooth"],
        "lambda_laplacian":    config["physics_engine"]["target_lambdas"]["laplacian"],
        "lambda_anticollapse": config["physics_engine"]["target_lambdas"]["anticollapse"],
        "lambda_homogeneity":  config["physics_engine"]["target_lambdas"]["homogeneity"],
        "lambda_body":         config["physics_engine"]["target_lambdas"]["body"],
        "lambda_bowel":        config["physics_engine"]["target_lambdas"]["bowel"],
        "lambda_femur":        config["physics_engine"]["target_lambdas"]["femur"],
        "lambda_penile":       config["physics_engine"]["target_lambdas"]["penile"],
        "lambda_global_ceil":  config["physics_engine"]["target_lambdas"]["global_ceil"],
        "lambda_bg":           config["physics_engine"]["target_lambdas"]["bg"],
    }

    print(f"\n{'='*60}")
    print(f"CURRICULUM RAMP: physics lambdas grow linearly over {WARMUP_EPOCHS} epochs")
    print(f"lambda_mse = 25.0  (constant throughout)")
    print(f"{'='*60}")
    logger.info(
        f"Curriculum ramp: lambda_mse=25.0 constant, "
        f"physics lambdas 0 -> target over {WARMUP_EPOCHS} epochs"
    )

    _lr     = config["training"]["learning_rate"]
    _lr_min = config["training"]["min_learning_rate"]
    _epochs = config["training"]["epochs"]
    optimizer = optim.Adam(model.parameters(), lr=_lr)

    epochs = _epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=_lr_min
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print(f"Model and dataloaders ready on {device}!")
    print(f"Epochs={epochs}  Scheduler=CosineAnnealingLR  "
          f"AMP={'ON' if torch.cuda.is_available() else 'OFF'}\n")
    logger.info(f"Model ready on {device}, epochs={epochs}, AMP={'ON' if torch.cuda.is_available() else 'OFF'}")

    # ---- Training --------------------------------------------------
    best_val_loss      = float("inf")   
    best_clinical_score = float("inf")  
    best_diagnostic_mse = float("inf")  

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']

        # ---- Lambda ramp (curriculum learning) ----------------------
        
        ramp_frac = min((epoch + 1) / WARMUP_EPOCHS, 1.0)
        for attr, target in PHYSICS_TARGET_LAMBDAS.items():
            setattr(loss_function, attr, target * ramp_frac)


        if epoch < WARMUP_EPOCHS:
            phase_tag = (f"RAMP {epoch + 1}/{WARMUP_EPOCHS} "
                         f"({100 * ramp_frac:.0f}% of physics)")
        else:
            phase_tag = "FULL PHYSICS"

        print(f"\nEpoch {epoch + 1}/{epochs}  [{phase_tag}]  (lr={current_lr:.2e})")
        logger.info(f"Epoch {epoch + 1}/{epochs} [{phase_tag}] ramp={ramp_frac:.3f} lr={current_lr:.2e}")

        # ---- Train -------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        step = 0

        accum_counter = 0  

        for batch in train_loader:
            step += 1
            accum_counter += 1
            
            inputs  = batch["image"].to(device, non_blocking=True)
            targets = batch["dose_label"].to(device, non_blocking=True)
            
            ring_mask_batch = batch["ring_mask"].to(device, non_blocking=True)
            
            bowel_mask_batch = batch["bowel_mask"].to(device, non_blocking=True)
            femur_mask_batch = batch["femur_mask"].to(device, non_blocking=True)

            targets = torch.nan_to_num(targets, nan=0.0)
            normalized_targets = targets / PRESCRIPTION_DOSE_GY

            ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

            if accum_counter == 1:
                optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
                dtype=torch.float16,
            ):
                outputs = model(inputs)

            # Prevent NaN propagation from mixed precision
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
            outputs_activated = F.softplus(outputs.float())

            body_mask_hard = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float()
            outputs_activated = outputs_activated * body_mask_hard

            loss, components = loss_function(
                outputs_activated,
                normalized_targets.float(),
                bladder_mask.float(),
                rectum_mask.float(),
                ptv_mask.float(),
                ring_mask_batch.float(),
                inputs,
                bowel_mask_batch.float(),
                femur_mask_batch.float(),
            )

            loss_scaled = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_scaled).backward()

            if accum_counter % GRAD_ACCUM_STEPS == 0:
                
                scaler.unscale_(optimizer)
                
                # Check for NaNs in gradients and clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                accum_counter = 0  

            train_loss_sum += loss.item()
            if step % 5 == 0 or step == len(train_loader):
                log_msg = (
                    f"Step {step}/{len(train_loader)} Loss={loss.item():.4f} "
                    f"mse={components['mse']:.4f} v_opt={components['v_opt']:.5f} "
                    f"v_mand={components['v_mand']:.5f} ptv={components['ptv']:.4f} "
                    f"ptv_max={components['ptv_max']:.5f} global_ceil={components['global_ceil']:.5f} "
                    f"ring={components['ring']:.5f} "
                    f"smooth={components['smooth']:.4f} laplacian={components['laplacian']:.4f} "
                    f"anticollapse={components['anticollapse']:.5f} "
                    f"shell_inner={components['shell_inner']:.5f} "
                    f"shell_outer={components['shell_outer']:.5f} "
                    f"homogeneity={components['homogeneity']:.4f} "
                    f"body={components['body']:.5f} "
                    f"bowel={components['bowel']:.5f} femur={components['femur']:.5f} "
                    f"penile={components['penile']:.5f} bg={components['bg']:.5f}"
                )
                print(f"  {log_msg}")
                logger.info(log_msg)

        if accum_counter % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss_avg = train_loss_sum / max(len(train_loader), 1)

        scheduler.step()

        # ---- Validation (skip if not every N epochs) -----------------
        val_loss_avg = float('nan')
        avg_ptv60_d95 = avg_ptv44_d95 = avg_bladder = avg_rectum = avg_dmax = avg_bg = avg_mse = avg_ring = float('nan')
        avg_ptv60 = avg_ptv44 = avg_bowel = avg_femur = avg_penile = float('nan')
        avg_hi = avg_ci = float('nan')

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_hi_sum = 0.0
            val_ci_sum = 0.0
            val_ptv60_d95_sum = 0.0   # D95 of PTV60 voxels only
            val_ptv44_d95_sum = 0.0   # D95 of PTV44 voxels only
            val_bladder_mean_sum = 0.0
            val_rectum_mean_sum = 0.0
            val_ring_mean_sum = 0.0
            val_dmax_sum = 0.0
            val_bg_mean_sum = 0.0
            val_ptv60_mean_sum = 0.0
            val_ptv44_mean_sum = 0.0
            val_bowel_mean_sum = 0.0
            val_femur_mean_sum = 0.0
            val_penile_mean_sum = 0.0
            n_val = 0
            n_val_ptv60 = 0
            n_val_ptv44 = 0

            logger.info(f"Running validation at epoch {epoch + 1}...")
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device)
                    targets = batch["dose_label"].to(device)

                    with torch.amp.autocast(
                        "cuda",
                        enabled=torch.cuda.is_available(),
                        dtype=torch.float16,
                    ):
                        outputs = sliding_window_inference(
                            inputs=inputs,
                            roi_size=PATCH_SIZE,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.25,
                        )

                    # Prevent NaN propagation in validation
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
                    outputs_activated = F.softplus(outputs.float())

                    body_mask_hard = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float()
                    outputs_activated = outputs_activated * body_mask_hard

                    normalized_targets = targets / PRESCRIPTION_DOSE_GY
                    ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)
                    
                    bowel_mask_val = batch["bowel_mask"].to(device)
                    femur_mask_val = batch["femur_mask"].to(device)

                    ring_mask_val = compute_ring_mask(ptv_mask)

                    # --- FIX FOR OOM: STRICT CPU OFFLOADING -------------------------------
                    mse_loss = ((outputs_activated - normalized_targets.float()) ** 2).mean()
                    val_loss_sum += mse_loss.item()
                    val_mse_sum += mse_loss.item()
                    
                    # 1. Immediately move outputs and masks to CPU
                    outputs_gy = (outputs_activated * PRESCRIPTION_DOSE_GY).cpu()
                    discrete_ptv = inputs[:, _PTV_CH_IDX:_PTV_CH_IDX+1, ...].cpu()
                    
                    ptv_mask_cpu = ptv_mask.cpu()
                    bladder_mask_cpu = bladder_mask.cpu()
                    rectum_mask_cpu = rectum_mask.cpu()
                    ring_mask_cpu = ring_mask_val.cpu()
                    bowel_mask_cpu = bowel_mask_val.cpu()
                    femur_mask_cpu = femur_mask_val.cpu()
                    body_mask_cpu = body_mask_hard.cpu()
                    
                    # 2. Slice and calculate means on CPU, extracting floats via .item()
                    # D95 is computed per-structure (no union blend) — dynamically fetching rx doses from config
                    ptv60_dose = outputs_gy[torch.isclose(discrete_ptv, torch.tensor(_PTV_PRIMARY_RX, device=discrete_ptv.device))]
                    if ptv60_dose.numel() > 0:
                        val_ptv60_d95_sum  += torch.quantile(ptv60_dose.float(), 0.05).item()
                        val_ptv60_mean_sum += ptv60_dose.mean().item()
                        
                        # -- ICRU 83 Homogeneity Index (HI) --
                        d2 = torch.quantile(ptv60_dose.float(), 0.98).item()
                        d98 = torch.quantile(ptv60_dose.float(), 0.02).item()
                        d50 = torch.median(ptv60_dose).item()
                        if d50 > 0:
                            val_hi_sum += (d2 - d98) / d50
                            
                        n_val_ptv60 += 1
                        
                    # -- RTOG Conformity Index (CI) --
                    v_ref = (outputs_gy >= _PTV_PRIMARY_RX).sum().item()
                    v_ptv = ptv60_dose.numel()
                    if v_ptv > 0:
                        val_ci_sum += v_ref / v_ptv
                        
                    ptv44_dose = outputs_gy[torch.isclose(discrete_ptv, torch.tensor(_PTV_SECONDARY_RX, device=discrete_ptv.device))]
                    if ptv44_dose.numel() > 0:
                        val_ptv44_d95_sum  += torch.quantile(ptv44_dose.float(), 0.05).item()
                        val_ptv44_mean_sum += ptv44_dose.mean().item()
                        n_val_ptv44 += 1
                        
                    # Penile Bulb from config-driven binary channel index
                    penile_mask_cpu = (inputs[:, _PENILE_CH_IDX:_PENILE_CH_IDX+1, ...] > 0.5).cpu()
                    penile_dose = outputs_gy[penile_mask_cpu.bool()]
                    if len(penile_dose) > 0: val_penile_mean_sum += penile_dose.mean().item()
                        
                    bowel_dose = outputs_gy[bowel_mask_cpu.bool()]
                    if len(bowel_dose) > 0: val_bowel_mean_sum += bowel_dose.mean().item()
                        
                    femur_dose = outputs_gy[femur_mask_cpu.bool()]
                    if len(femur_dose) > 0: val_femur_mean_sum += femur_dose.mean().item()

                    bladder_dose = outputs_gy[bladder_mask_cpu.bool()]
                    if len(bladder_dose) > 0: val_bladder_mean_sum += bladder_dose.mean().item()

                    rectum_dose = outputs_gy[rectum_mask_cpu.bool()]
                    if len(rectum_dose) > 0: val_rectum_mean_sum += rectum_dose.mean().item()

                    ring_dose = outputs_gy[ring_mask_cpu.bool()]
                    if len(ring_dose) > 0: val_ring_mean_sum += ring_dose.mean().item()

                    val_dmax_sum += outputs_gy.max().item()

                    bg_mask = (body_mask_cpu.bool() & ~ptv_mask_cpu.bool() & 
                               ~bladder_mask_cpu.bool() & ~rectum_mask_cpu.bool() & 
                               ~bowel_mask_cpu.bool() & ~femur_mask_cpu.bool() &
                               ~ring_mask_cpu.bool())
                    bg_dose = outputs_gy[bg_mask]
                    if len(bg_dose) > 0: val_bg_mean_sum += bg_dose.mean().item()

                    n_val += 1

                    # 3. Aggressively delete CPU tensors to free system RAM
                    del outputs_gy, discrete_ptv, ptv60_dose, ptv44_dose, bowel_dose, femur_dose, penile_dose, bladder_dose, rectum_dose, ring_dose, bg_dose
                    del ptv_mask_cpu, bladder_mask_cpu, rectum_mask_cpu, ring_mask_cpu
                    del bowel_mask_cpu, femur_mask_cpu, body_mask_cpu, bg_mask, penile_mask_cpu
                    
                    del outputs, outputs_activated, body_mask_hard
                    del ptv_mask, bladder_mask, rectum_mask, ring_mask_val
                    del inputs, targets, normalized_targets
                    del bowel_mask_val, femur_mask_val
                    torch.cuda.empty_cache()

            val_loss_avg = val_loss_sum / max(n_val, 1)
            avg_mse = val_mse_sum / max(n_val, 1)
            avg_ptv60_d95 = val_ptv60_d95_sum / max(n_val_ptv60, 1)
            avg_ptv44_d95 = val_ptv44_d95_sum / max(n_val_ptv44, 1)
            avg_bladder = val_bladder_mean_sum / max(n_val, 1)
            avg_rectum = val_rectum_mean_sum / max(n_val, 1)
            avg_ring = val_ring_mean_sum / max(n_val, 1)
            avg_dmax = val_dmax_sum / max(n_val, 1)
            avg_bg = val_bg_mean_sum / max(n_val, 1)
            avg_hi = val_hi_sum / max(n_val_ptv60, 1)
            avg_ci = val_ci_sum / max(n_val, 1)
            avg_ptv60 = val_ptv60_mean_sum / max(n_val_ptv60, 1)
            avg_ptv44 = val_ptv44_mean_sum / max(n_val_ptv44, 1)
            avg_bowel = val_bowel_mean_sum / max(n_val, 1)
            avg_femur = val_femur_mean_sum / max(n_val, 1)
            avg_penile = val_penile_mean_sum / max(n_val, 1)

        epoch_summary = (
            f"Epoch {epoch + 1} Summary: Train={train_loss_avg:.4f} Val={val_loss_avg:.4f} (MSE={avg_mse:.4f})\n"
            f"          Coverage D95:  {_PTV_PRIMARY_NAME}_D95={avg_ptv60_d95:.2f}Gy  {_PTV_SECONDARY_NAME}_D95={avg_ptv44_d95:.2f}Gy\n"
            f"          SIB Means:    {_PTV_PRIMARY_NAME}={avg_ptv60:.2f}Gy {_PTV_SECONDARY_NAME}={avg_ptv44:.2f}Gy\n"
            f"          OAR Metrics:  Bladder={avg_bladder:.2f}Gy Rectum={avg_rectum:.2f}Gy Bowel={avg_bowel:.2f}Gy Femur={avg_femur:.2f}Gy Ring={avg_ring:.2f}Gy Penile={avg_penile:.2f}Gy\n"
            f"          Physics:      Dmax={avg_dmax:.2f}Gy BG={avg_bg:.2f}Gy HI={avg_hi:.3f} CI={avg_ci:.2f}"
        )
        print(f"  --> {epoch_summary}")
        logger.info(epoch_summary)

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            
            is_physically_valid = (avg_dmax < 80.0)

            if is_physically_valid and val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(model.state_dict(), "best_dose_model_physics_jun3.pth")
                checkpoint_msg = f"[PHYSICS] Saved best model val_loss={best_val_loss:.4f}"
                print(f"  --> {checkpoint_msg}")
                logger.info(checkpoint_msg)

            current_worst_oar = max(avg_bladder, avg_rectum)
            ptv_deficit = max(0.0, _PTV_PRIMARY_RX - avg_ptv60_d95)
            clinical_score = current_worst_oar + (ptv_deficit * config["scoring"]["clinical_ptv_weight"])

            if is_physically_valid and clinical_score < best_clinical_score:
                best_clinical_score = clinical_score
                torch.save(model.state_dict(), "best_dose_model_clinical_jun3.pth")
                clinical_msg = (
                    f"[CLINICAL] Saved best model Score={clinical_score:.3f} "
                    f"{_PTV_PRIMARY_NAME}_D95={avg_ptv60_d95:.2f}Gy {_PTV_SECONDARY_NAME}_D95={avg_ptv44_d95:.2f}Gy "
                    f"Bladder={avg_bladder:.2f}Gy Rectum={avg_rectum:.2f}Gy"
                )
                print(f"  --> {clinical_msg}")
                logger.info(clinical_msg)

            if val_loss_avg < best_diagnostic_mse:
                best_diagnostic_mse = val_loss_avg
                torch.save(model.state_dict(), "best_dose_model_diagnostic_jun3.pth")
                diag_msg = f"[DIAGNOSTIC] Saved fallback model MSE={best_diagnostic_mse:.4f}"
                print(f"  --> {diag_msg}")
                logger.info(diag_msg)

    completion_msg = (
        f"Training complete. Best val loss: {best_val_loss:.4f} "
        f"Best clinical score: {best_clinical_score:.4f}"
    )
    print(f"\n{completion_msg}")
    logger.info(completion_msg)

    # ====================================================================
    
    # ====================================================================
    print("\n" + "=" * 68)
    print("FINAL CLINICAL EVALUATION  (Physics + Clinical checkpoints)")
    print("=" * 68)

    # ---- Imports needed only for this block ----------------------------
    import pandas as pd  # pyrefly: ignore [missing-import]

    # ---- Physical dose ceiling (Gy) — hard clinical constraint ---------
    PHYSICAL_MAX_GY = config["clinical_targets"]["physical_max_gy"]

    # ---- DVH helpers ---------------------------------------------------
    def quantile_dose(dose_1d: torch.Tensor, pct: float) -> float:
        """
        Return the dose (Gy) exceeded by `pct`% of the volume.
        D95 = dose exceeded by 95% of voxels  ->  quantile at 0.05 tail.
        """
        if dose_1d.numel() == 0:
            return float("nan")
        q = 1.0 - pct / 100.0
        return torch.quantile(dose_1d.float(), q).item()

    def v_metric(dose_1d: torch.Tensor, threshold_gy: float) -> float:
        """
        Volume fraction (%) of organ receiving > threshold_gy Gy.
        """
        if dose_1d.numel() == 0:
            return float("nan")
        return ((dose_1d > threshold_gy).float().mean() * 100.0).item()

    # ---- Dual-model evaluation loop ------------------------------------
    for model_path, csv_name in [
        ("best_dose_model_physics_jun3.pth",    "validation_physics_summary_jun3.csv"),
        ("best_dose_model_clinical_jun3.pth",   "validation_clinical_summary_jun3.csv"),
        ("best_dose_model_diagnostic_jun3.pth", "validation_diagnostic_summary_jun3.csv"),
    ]:
        print(f"\n--- Evaluating: {model_path} -> {csv_name} ---")

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  Loaded weights from '{model_path}'")
        else:
            print(f"  WARNING: '{model_path}' not found — skipping.")
            continue

        model.eval()
        records = []

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                inputs  = batch["image"].to(device)
                targets = batch["dose_label"].to(device)

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                    dtype=torch.float16,
                ):
                    outputs = sliding_window_inference(
                        inputs=inputs,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                    )

                outputs_gy = F.softplus(outputs.float()) * PRESCRIPTION_DOSE_GY
                outputs_gy = torch.clamp(outputs_gy, min=0.0, max=PHYSICAL_MAX_GY)

                # Apply body mask to prevent ghost radiation inflating CSV metrics
                body_mask_eval = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float()
                outputs_gy = outputs_gy * body_mask_eval

                ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)
                penile_mask_eval = (inputs[:, _PENILE_CH_IDX:_PENILE_CH_IDX+1, ...] > 0.5)

                discrete_ptv = inputs[:, _PTV_CH_IDX:_PTV_CH_IDX+1, ...].cpu()
                
                # SIB Mapping from CreateDiscretePTVMapd
                sib_eval_targets = {
                    lvl["name"]: lvl["rx_gy"]
                    for lvl in config["clinical_targets"]["targets"]
                }
                bladder_dose = outputs_gy[bladder_mask.bool()].cpu()
                rectum_dose  = outputs_gy[rectum_mask.bool()].cpu()
                penile_dose  = outputs_gy[penile_mask_eval.bool()].cpu()

                try:
                    label_path = val_files[idx]["dose_label"]
                    patient_id = os.path.basename(label_path).replace(".nii.gz", "")
                except (IndexError, KeyError):
                    patient_id = f"patient_{idx:03d}"

                row = {"Patient_ID": patient_id}

                # Calculate metrics for each specific SIB target
                for name, id_val in sib_eval_targets.items():
                    mask = torch.isclose(discrete_ptv, torch.tensor(id_val, dtype=torch.float32))
                    sib_dose = outputs_gy[mask.bool()]
                    
                    if sib_dose.numel() > 0:
                        row[f"{name}_D95 (Gy)"] = quantile_dose(sib_dose, 95)
                        row[f"{name}_Mean (Gy)"] = sib_dose.mean().item()
                        row[f"{name}_Max (Gy)"] = sib_dose.max().item()
                    else:
                        row[f"{name}_D95 (Gy)"] = float("nan")
                        row[f"{name}_Mean (Gy)"] = float("nan")
                        row[f"{name}_Max (Gy)"] = float("nan")

                row["Bladder_Mean (Gy)"] = bladder_dose.mean().item() if bladder_dose.numel() else float("nan")
                row["Bladder_Max (Gy)"]  = bladder_dose.max().item()  if bladder_dose.numel() else float("nan")
                for thresh in config["evaluation"]["oar_v_metrics_gy"]:
                    row[f"Bladder_V{thresh}Gy (%)"] = v_metric(bladder_dose, thresh)

                row["Rectum_Mean (Gy)"] = rectum_dose.mean().item() if rectum_dose.numel() else float("nan")
                row["Rectum_Max (Gy)"]  = rectum_dose.max().item()  if rectum_dose.numel() else float("nan")
                for thresh in config["evaluation"]["oar_v_metrics_gy"]:
                    row[f"Rectum_V{thresh}Gy (%)"] = v_metric(rectum_dose, thresh)

                row["PenileBulb_Mean (Gy)"] = penile_dose.mean().item() if penile_dose.numel() else float("nan")
                row["PenileBulb_V47Gy (%)"] = v_metric(penile_dose, 47.0)  # v3: V47Gy ≤ 50%

                records.append(row)
                print(
                    f"  [{idx + 1}/{len(val_loader)}] {patient_id}  "
                    f"PTV60 D95={row.get('PTV60_D95 (Gy)', float('nan')):.2f} Gy  "
                    f"Bladder Mean={row['Bladder_Mean (Gy)']:.2f} Gy  "
                    f"Rectum Mean={row['Rectum_Mean (Gy)']:.2f} Gy  "
                    f"PenileBulb Mean={row['PenileBulb_Mean (Gy)']:.2f} Gy"
                )

        df = pd.DataFrame(records)
        float_cols = [c for c in df.columns if c != "Patient_ID"]
        df[float_cols] = df[float_cols].round(2)
        df.to_csv(csv_name, index=False)
        print(f"\n  Saved '{csv_name}'")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
