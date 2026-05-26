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

# ===================================================================

# ===================================================================
DATA_DIR = os.environ.get("DATA_DIR", "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

CHANNELS = ["0000", "0001", "0002", "0003", "0004", "0005"]  
TARGET_SPACING = (1.27, 1.27, 2.5)
PATCH_SIZE = (128, 128, 64)  

PRESCRIPTION_DOSE_GY = 75.0  
CONSTRAINT_CSV = os.environ.get(
    "CONSTRAINT_CSV", "./prostate_prime_constraints_v2.csv"
)

GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))

VAL_EVERY_N_EPOCHS = int(os.environ.get("VAL_EVERY_N_EPOCHS", "1"))  

WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "75"))

# ===================================================================

# ===================================================================

def load_clinical_constraints(csv_path, patient_class="N0"):
    """
    Parse prostate_prime_constraints_v2.csv into structured dicts.

    V2 schema (one row per tier):
      Name               — structure name; plain = N0, _Nplus suffix = N+
      Type               — PTV / CTV / Avoidance
      Constraint_Type    — D or V
      Constraint_Value   — numeric dose threshold (Gy) or percentile string
      Constraint_Unit    — Gy or %
      Constraint_Priority
      Evaluation_Type    — <=, >=, <, >
      Objective_Value    — the limit value (volume fraction or dose fraction)
      Objective_Unit     — % or Gy or cc
      Objective_Type     — "Optimal" or "Mandatory"

    Patient class mapping:
      N0  -> Name in {"Bladder", "Anorectum", "PTV62", "PTV44", "PTV55", ...}
      N+  -> Name ends with "_Nplus"

    Returns
    -------
    dict  with keys:
        "v_type"  -> {"Bladder": [...], "Anorectum": [...]}
        "d_type"  -> {"PTV_max_dose_gy": float or None,
                      "PTV_coverage": [...]}
    All V-Type dose thresholds are normalised to [0, 1] by dividing
    by PRESCRIPTION_DOSE_GY (75 Gy).
    """
    
    v_accum = {}          
    ptv_coverage = []
    ptv_max_dose_gy = None

    nplus_suffix = "_Nplus"

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

            # ---- Determine patient class from Name ------------------
            is_nplus = name.endswith(nplus_suffix)
            if patient_class == "N0" and is_nplus:
                continue
            if patient_class == "N+" and not is_nplus:
                continue

            canonical = name[: -len(nplus_suffix)] if is_nplus else name

            # ---- V-Type constraints (Bladder / Anorectum) -----------
            if ctype == "V" and canonical in ("Bladder", "Anorectum"):
                
                if obj_unit.strip() != "%":
                    continue  
                
                dose_thresh_gy = float(c_val_raw)
                obj_value      = float(obj_val_raw)

                key = (canonical, dose_thresh_gy)
                if key not in v_accum:
                    v_accum[key] = {"optimal_v": float("nan"),
                                    "mandatory_v": float("nan")}

                if obj_type == "Optimal":
                    v_accum[key]["optimal_v"] = obj_value
                elif obj_type == "Mandatory":
                    v_accum[key]["mandatory_v"] = obj_value

            # ---- D-Type: PTV max dose -------------------------------
            if (ctype == "D" and struct_type == "PTV"
                    and c_val_raw == "Max" and c_unit == "Gy"):
                if obj_type == "Mandatory":
                    ptv_max_dose_gy = float(obj_val_raw)

            # ---- D-Type: PTV coverage (D95 / D98 etc.) -------------
            
            if (ctype == "D" and struct_type == "PTV"
                    and c_unit == "%" and obj_unit == "%"):
                try:
                    percentile = float(c_val_raw)
                except ValueError:
                    continue
                if percentile >= 90 and obj_type == "Mandatory":
                    ptv_coverage.append(
                        {
                            "metric": f"D{int(percentile)}",
                            "fraction": float(obj_val_raw),  
                        }
                    )

    # ---- Build the final v_constraints list (per organ) -------------
    v_constraints = {"Bladder": [], "Anorectum": []}
    for (organ, dose_gy), tiers in sorted(v_accum.items(),
                                          key=lambda x: x[0][1]):
        
        if math.isnan(tiers["mandatory_v"]):
            continue
        norm_dose = dose_gy / PRESCRIPTION_DOSE_GY
        v_constraints[organ].append(
            {
                "dose_gy":     dose_gy,
                "norm_dose":   norm_dose,
                "optimal_v":   tiers["optimal_v"],
                "mandatory_v": tiers["mandatory_v"],
            }
        )

    return {
        "v_type": v_constraints,
        "d_type": {
            "PTV_max_dose_gy": ptv_max_dose_gy,
            "PTV_coverage":    ptv_coverage,
        },
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
        lambda_beam=5.0,            
        lambda_ptv_max=150.0,       
        lambda_homogeneity=30.0,    
        lambda_laplacian=5.0,       
        lambda_bowel=15.0,          
        lambda_femur=10.0,          
        lambda_global_ceil=2.0,     
        lambda_beam_ceil=2.0,       
        lambda_shell_inner=0.0,     
        lambda_shell_outer=0.0,     
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
        self.lambda_beam = lambda_beam
        self.lambda_ptv_max = lambda_ptv_max
        self.lambda_homogeneity = lambda_homogeneity
        self.lambda_laplacian = lambda_laplacian
        self.lambda_bowel = lambda_bowel
        self.lambda_femur = lambda_femur
        self.lambda_global_ceil = lambda_global_ceil
        self.lambda_beam_ceil = lambda_beam_ceil
        self.lambda_shell_inner = lambda_shell_inner
        self.lambda_shell_outer = lambda_shell_outer
        self.lambda_body = 20.0    
        self.k = k_steepness

    # --- Differentiable DVH volume fraction --------------------------
    def calculate_dvh_volume(self, predicted_dose, organ_mask, norm_dose_threshold):
        """
        V^pred_{D_ref} = (1/N_OAR) * Σ_{i∈OAR} σ(k·(D_i - D_ref))

        Uses torch.sigmoid — fully differentiable.
        organ_mask: binary float tensor, same spatial shape as predicted_dose.
        """
        
        organ_voxels = predicted_dose * organ_mask         
        n_organ = organ_mask.sum()

        if n_organ < 1.0:
            return torch.tensor(0.0, device=predicted_dose.device,
                                dtype=predicted_dose.dtype)

        step_approx = torch.sigmoid(self.k * (organ_voxels - norm_dose_threshold))
        
        step_approx = step_approx * organ_mask
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
            "Bladder": bladder_mask,
            "Anorectum": rectum_mask,
        }

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
        loss_ptv = torch.tensor(0.0, device=pred_dose.device,
                                dtype=pred_dose.dtype)
        ptv_n = ptv_mask.sum()
        if ptv_n > 0:
            for rule in self.constraints["d_type"]["PTV_coverage"]:
                frac = rule["fraction"]
                underdose = torch.relu(frac - pred_dose) * ptv_mask
                loss_ptv = loss_ptv + (underdose ** 2).sum() / ptv_n

        # ------ 3a. Homogeneity Penalty (Target exactly 62.4 Gy) ----
        TARGET_GY = 62.4
        target_norm = TARGET_GY / PRESCRIPTION_DOSE_GY
        loss_homogeneity = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 0:
            deviation = (pred_dose - target_norm) * ptv_mask
            loss_homogeneity = (deviation ** 2).sum() / ptv_n

        # ------ 3b. Anti-Collapse Safety Net ------------------------
        
        loss_anticollapse = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 0:
            ptv_underdose_frac = (torch.relu(0.50 - pred_dose) * ptv_mask).sum() / ptv_n
            loss_anticollapse = ptv_underdose_frac ** 2

        # ------ 4. L_D-Type max dose (PTV max) ----------------------
        # ------ 4. L_D-Type max dose (Hotspot Smasher) --------------
        
        HARD_MAX_GY = 66.34
        hard_max_norm = HARD_MAX_GY / PRESCRIPTION_DOSE_GY
        
        loss_ptv_max = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 0:
            
            overdose = torch.relu(pred_dose - hard_max_norm) * ptv_mask
            loss_ptv_max = (overdose ** 2).sum() / ptv_n.clamp(min=1.0)

        # ------ Global Hard Ceiling (Top-K L1 formulation) --------------
        
        global_ceil_norm = 72.0 / 75.0
        body_mask_bool = inputs[:, 5:6, ...] > 0.5
        body_pred = pred_dose[body_mask_bool]

        loss_global_ceil = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        
        if body_pred.numel() > 0:
            ceil_violations = torch.relu(body_pred - global_ceil_norm)
            
            if ceil_violations.max() > 0:
                
                K = max(int(0.001 * body_pred.numel()), 10)
                K = min(K, body_pred.numel())  
                
                topk_violations, _ = torch.topk(ceil_violations, K)
                
                loss_global_ceil = topk_violations.sum()

        # ------ 5. L_Ring (Falloff shell penalty) -------------------
        
        RING_THRESH = 0.88
        ring_n = ring_mask.sum()
        if ring_n > 0:
            ring_overdose = torch.relu(pred_dose - RING_THRESH) * ring_mask
            loss_ring = (ring_overdose ** 2).sum() / ring_n
        else:
            loss_ring = torch.tensor(0.0, device=pred_dose.device,
                                     dtype=pred_dose.dtype)

        # ------ 5e. Concentric Shell Falloff Penalties ---------------
        
        ptv_for_dilation = ptv_mask.float()  

        dil_20mm = F.max_pool3d(
            ptv_for_dilation,
            kernel_size=(17, 33, 33),
            stride=1,
            padding=(8, 16, 16),
        )
        dil_40mm = F.max_pool3d(
            ptv_for_dilation,
            kernel_size=(33, 63, 63),
            stride=1,
            padding=(16, 31, 31),
        )

        shell_inner_mask = torch.clamp(dil_20mm - ptv_for_dilation, 0.0, 1.0)
        shell_outer_mask = torch.clamp(dil_40mm - dil_20mm, 0.0, 1.0)

        oar_exclusion = torch.clamp(
            bladder_mask + rectum_mask, 0.0, 1.0
        )
        shell_inner_mask = shell_inner_mask * (1.0 - oar_exclusion)
        shell_outer_mask = shell_outer_mask * (1.0 - oar_exclusion)

        SHELL_INNER_CEIL = 45.0 / PRESCRIPTION_DOSE_GY  
        loss_shell_inner = torch.tensor(0.0, device=pred_dose.device,
                                        dtype=pred_dose.dtype)
        inner_pred = pred_dose[shell_inner_mask.bool()]
        if inner_pred.numel() > 0:
            inner_violations = torch.relu(inner_pred - SHELL_INNER_CEIL)
            if inner_violations.max() > 0:
                K_inner = max(int(0.001 * inner_pred.numel()), 10)
                K_inner = min(K_inner, inner_pred.numel())
                topk_inner, _ = torch.topk(inner_violations, K_inner)
                loss_shell_inner = topk_inner.sum()

        SHELL_OUTER_CEIL = 30.0 / PRESCRIPTION_DOSE_GY  
        loss_shell_outer = torch.tensor(0.0, device=pred_dose.device,
                                        dtype=pred_dose.dtype)
        outer_pred = pred_dose[shell_outer_mask.bool()]
        if outer_pred.numel() > 0:
            outer_violations = torch.relu(outer_pred - SHELL_OUTER_CEIL)
            if outer_violations.max() > 0:
                K_outer = max(int(0.001 * outer_pred.numel()), 10)
                K_outer = min(K_outer, outer_pred.numel())
                topk_outer, _ = torch.topk(outer_violations, K_outer)
                loss_shell_outer = topk_outer.sum()

        # ------ 5b. L_Beam (Anti-Brachytherapy Suppression) --------------
        beam_mask = inputs[:, 4:5, ...]
        body_mask = (inputs[:, 5:6, ...] > 0.5).float()
        
        exclusion_mask = torch.clamp(ptv_mask + ring_mask + bladder_mask + rectum_mask, 0.0, 1.0)
        outside_beam_body = (1.0 - beam_mask) * body_mask * (1.0 - exclusion_mask)
        rogue_dose = pred_dose * outside_beam_body
        n_rogue = outside_beam_body.sum().clamp(min=1.0)
        loss_beam_suppression = rogue_dose.sum() / n_rogue

        # ------ 5d. Beam Corridor Hard Ceiling (Top-K L1 Sum) --------------
        
        beam_interior_mask = beam_mask * body_mask * (1.0 - ptv_mask) *                             (1.0 - ring_mask) * (1.0 - bladder_mask) * (1.0 - rectum_mask)
        beam_pred = pred_dose[beam_interior_mask.bool()]
        
        loss_beam_ceil = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if beam_pred.numel() > 0:
            beam_ceil_violations = torch.relu(beam_pred - 0.60) 
            if beam_ceil_violations.max() > 0:
                
                K_beam = max(int(0.001 * beam_pred.numel()), 10)
                K_beam = min(K_beam, beam_pred.numel()) 
                
                topk_beam, _ = torch.topk(beam_ceil_violations, K_beam)
                
                loss_beam_ceil = topk_beam.sum()

        # ------ 5c. L_Body (Anti-Ghost Suppression) ----------------------
        
        body_mask = (inputs[:, 5:6, ...] > 0.5).float()
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
        
        BOWEL_OPT_THRESH  = 45.0 / PRESCRIPTION_DOSE_GY   
        BOWEL_OPT_LIMIT   = 0.30
        BOWEL_MAND_THRESH = 50.0 / PRESCRIPTION_DOSE_GY   
        BOWEL_MAND_LIMIT  = 0.50
        MANDATORY_SCALE   = 5.0   

        bowel_v45 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_OPT_THRESH)
        bowel_v50 = self.calculate_dvh_volume(pred_dose, bowel_mask, BOWEL_MAND_THRESH)
        loss_bowel_opt  = torch.relu(bowel_v45 - BOWEL_OPT_LIMIT)  ** 2
        loss_bowel_mand = torch.relu(bowel_v50 - BOWEL_MAND_LIMIT) ** 2
        loss_bowel = loss_bowel_opt + MANDATORY_SCALE * loss_bowel_mand

        # ------ 8. L_Femur (merged Femur_Head_L+R — dual-tier) ------
        
        FEMUR_THRESH_NORM = 50.0 / PRESCRIPTION_DOSE_GY   
        FEMUR_OPT_LIMIT   = 0.05
        FEMUR_MAND_LIMIT  = 0.50

        femur_v50 = self.calculate_dvh_volume(pred_dose, femur_mask, FEMUR_THRESH_NORM)
        loss_femur_opt  = torch.relu(femur_v50 - FEMUR_OPT_LIMIT)  ** 2
        loss_femur_mand = torch.relu(femur_v50 - FEMUR_MAND_LIMIT) ** 2
        loss_femur = loss_femur_opt + MANDATORY_SCALE * loss_femur_mand

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
            + self.lambda_beam       * loss_beam_suppression
            + self.lambda_beam_ceil   * loss_beam_ceil
            + self.lambda_shell_inner * loss_shell_inner
            + self.lambda_shell_outer * loss_shell_outer
            + self.lambda_homogeneity * loss_homogeneity
            + self.lambda_body        * loss_body
            + self.lambda_bowel       * loss_bowel
            + self.lambda_femur       * loss_femur
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
            "beam":       loss_beam_suppression.item(),
            "beam_ceil":    loss_beam_ceil.item(),
            "shell_inner":  loss_shell_inner.item(),
            "shell_outer":  loss_shell_outer.item(),
            "homogeneity":  loss_homogeneity.item(),
            "body":       loss_body.item(),
            "bowel":      loss_bowel.item(),
            "femur":      loss_femur.item(),
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
        
        pt_dict["bowel_mask"] = os.path.join(IMAGES_DIR, f"{patient_id}_bowel.nii.gz")
        pt_dict["femur_mask"] = os.path.join(IMAGES_DIR, f"{patient_id}_femur.nii.gz")
        data_dicts.append(pt_dict)
    return data_dicts

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

    def __init__(self, keys, body_thresh_hu: float = -300.0,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.body_thresh_hu = body_thresh_hu

    def __call__(self, data):
        d = dict(data)
        
        ct      = d["ch_0"]           
        ptv     = d["ch_1"] >= 0.5    
        bladder = d["ch_2"] <= 0.0    
        rectum  = d["ch_3"] <= 0.0    
        body    = ct >= self.body_thresh_hu   

        crop_mask = torch.zeros_like(ct, dtype=torch.float32)
        
        crop_mask[body]    = 1.0   
        crop_mask[ptv]     = 2.0   
        crop_mask[bladder] = 3.0   
        crop_mask[rectum]  = 4.0   

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

ALL_KEYS = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5",
            "bowel_mask", "femur_mask",  
            "dose_label"]

train_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest",
                  "nearest", "nearest",   
                  "bilinear"),             
        ),
        
        Create5ClassCropMaskd(keys=["ch_0"], body_thresh_hu=-300.0),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        
        CreateFalloffRingd(keys=["ch_1"]),
        
        RandCropByLabelClassesd(
            keys=ALL_KEYS + ["ring_mask"],   
            label_key="crop_mask",
            spatial_size=PATCH_SIZE,
            num_classes=5,
            ratios=[0.0, 1.0, 1.0, 1.0, 1.0],
            num_samples=2,  
        ),
        DeleteItemsd(keys=["crop_mask"]),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5"], name="image"),
        ToTensord(keys=["image", "dose_label", "ring_mask", "bowel_mask", "femur_mask"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "nearest", "nearest",
                  "nearest", "nearest",   
                  "bilinear"),             
        ),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "ch_5"], name="image"),
        ToTensord(keys=["image", "dose_label", "bowel_mask", "femur_mask"]),
    ]
)

# ===================================================================

# ===================================================================

def extract_binary_masks(inputs):
    """
    Convert the concatenated input tensor channels to binary masks.
    Channels: 0=CT, 1=PTV (binary), 2=Bladder SDM, 3=Anorectum SDM.

    SDMs are <= 0.0 inside the organ.  We convert to binary float masks.
    PTV channel is already binary (>= 0.5 → 1.0).

    Returns ptv_mask, bladder_mask, rectum_mask  — all (B,1,D,H,W) float.
    """
    ptv_mask = (inputs[:, 1:2, ...] >= 0.5).float()
    bladder_mask = (inputs[:, 2:3, ...] <= 0.0).float()
    rectum_mask = (inputs[:, 3:4, ...] <= 0.0).float()
    return ptv_mask, bladder_mask, rectum_mask

# ===================================================================

# ===================================================================

def main():
    # ---- Setup logging ---------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
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
    logger.info(f"Loading clinical constraints from {CONSTRAINT_CSV} ...")
    print(f"Loading clinical constraints from {CONSTRAINT_CSV} ...")
    constraints = load_clinical_constraints(CONSTRAINT_CSV, patient_class="N0")

    n_bladder = len(constraints["v_type"]["Bladder"])
    n_rect = len(constraints["v_type"]["Anorectum"])
    n_ptv = len(constraints["d_type"]["PTV_coverage"])
    ptv_max = constraints["d_type"]["PTV_max_dose_gy"]
    print(f"  V-Type constraints:  Bladder={n_bladder}  Anorectum={n_rect}")
    print(f"  D-Type constraints:  PTV coverage rules={n_ptv}  PTV max={ptv_max} Gy")

    for organ in ("Bladder", "Anorectum"):
        for r in constraints["v_type"][organ]:
            print(f"    {organ}  V{r['dose_gy']}Gy  "
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

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
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
        in_channels=6,  
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
        lambda_beam=0.0,            
        lambda_ptv_max=0.0,         
        lambda_homogeneity=0.0,     
        lambda_global_ceil=2.0,     
        lambda_beam_ceil=2.0,       
        lambda_shell_inner=0.0,     
        lambda_shell_outer=0.0,     
        lambda_bowel=0.0,           
        lambda_femur=0.0,           
        k_steepness=50.0,
    )
    loss_function.lambda_body = 0.0  

    PHYSICS_TARGET_LAMBDAS = {
        "lambda_optional":      2.0,
        "lambda_mandatory":    50.0,
        "lambda_ptv":          15.0,
        "lambda_ptv_max":      150.0,
        "lambda_ring":         30.0,
        "lambda_smooth":        1.0,
        "lambda_laplacian":     5.0,
        "lambda_anticollapse": 150.0,
        "lambda_beam":         25.0,
        "lambda_shell_inner":   2.0,   
        "lambda_shell_outer":   2.0,   
        "lambda_homogeneity":  30.0,
        "lambda_body":         20.0,
        "lambda_bowel":        15.0,  
        "lambda_femur":        10.0,  
    }

    print(f"\n{'='*60}")
    print(f"CURRICULUM RAMP: physics lambdas grow linearly over {WARMUP_EPOCHS} epochs")
    print(f"lambda_mse = 25.0  (constant throughout)")
    print(f"{'='*60}")
    logger.info(
        f"Curriculum ramp: lambda_mse=25.0 constant, "
        f"physics lambdas 0 -> target over {WARMUP_EPOCHS} epochs"
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print(f"Model and dataloaders ready on {device}!")
    print(f"Epochs={epochs}  Scheduler=CosineAnnealingLR  "
          f"AMP={'ON' if torch.cuda.is_available() else 'OFF'}\n")
    logger.info(f"Model ready on {device}, epochs={epochs}, AMP={'ON' if torch.cuda.is_available() else 'OFF'}")

    # ---- Training --------------------------------------------------
    best_val_loss      = float("inf")   

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']

        # ---- Lambda ramp (curriculum learning) ----------------------
        
        ramp_frac = min((epoch + 1) / WARMUP_EPOCHS, 1.0)
        for attr, target in PHYSICS_TARGET_LAMBDAS.items():
            setattr(loss_function, attr, target * ramp_frac)

        # ---- Ceiling lambda partial ramp (first 20 epochs only) -----
        
        CEIL_RAMP_EPOCHS = 20
        ceil_ramp = min((epoch + 1) / CEIL_RAMP_EPOCHS, 1.0)
        loss_function.lambda_global_ceil = 2.0 * ceil_ramp
        loss_function.lambda_beam_ceil   = 2.0 * ceil_ramp

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

            outputs_activated = F.softplus(outputs.float())

            body_mask_hard = (inputs[:, 5:6, ...] > 0.5).float()
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
                    f"beam={components['beam']:.5f} beam_ceil={components['beam_ceil']:.5f} "
                    f"shell_inner={components['shell_inner']:.5f} "
                    f"shell_outer={components['shell_outer']:.5f} "
                    f"homogeneity={components['homogeneity']:.4f} "
                    f"body={components['body']:.5f} "
                    f"bowel={components['bowel']:.5f} femur={components['femur']:.5f}"
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
        avg_d95 = avg_bladder = avg_rectum = avg_dmax = avg_bg = avg_mse = avg_ring = float('nan')

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_d95_sum = 0.0
            val_bladder_mean_sum = 0.0
            val_rectum_mean_sum = 0.0
            val_ring_mean_sum = 0.0
            val_dmax_sum = 0.0
            val_bg_mean_sum = 0.0
            n_val = 0

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

                    outputs_activated = F.softplus(outputs.float())

                    body_mask_hard = (inputs[:, 5:6, ...] > 0.5).float()
                    outputs_activated = outputs_activated * body_mask_hard

                    normalized_targets = targets / PRESCRIPTION_DOSE_GY
                    ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)
                    
                    bowel_mask_val = batch["bowel_mask"].to(device)
                    femur_mask_val = batch["femur_mask"].to(device)

                    ring_mask_val = compute_ring_mask(ptv_mask)

                    # --- FIX FOR OOM: -----------------------------------------------------
                    
                    mse_loss = ((outputs_activated - normalized_targets.float()) ** 2).mean()
                    
                    val_loss_sum += mse_loss.item()
                    val_mse_sum += mse_loss.item()

                    outputs_gy = outputs_activated * PRESCRIPTION_DOSE_GY

                    ptv_dose = outputs_gy[ptv_mask.bool()]
                    if len(ptv_dose) > 0:
                        val_d95_sum += torch.quantile(ptv_dose, 0.05).item()

                    bladder_dose = outputs_gy[bladder_mask.bool()]
                    if len(bladder_dose) > 0:
                        val_bladder_mean_sum += bladder_dose.mean().item()

                    rectum_dose = outputs_gy[rectum_mask.bool()]
                    if len(rectum_dose) > 0:
                        val_rectum_mean_sum += rectum_dose.mean().item()

                    ring_dose = outputs_gy[ring_mask_val.bool()]
                    if len(ring_dose) > 0:
                        val_ring_mean_sum += ring_dose.mean().item()

                    val_dmax_sum += outputs_gy.max().item()

                    bg_mask = (body_mask_hard.bool() & ~ptv_mask.bool() & 
                               ~bladder_mask.bool() & ~rectum_mask.bool() & 
                               ~bowel_mask_val.bool() & ~femur_mask_val.bool())
                    bg_dose = outputs_gy[bg_mask]
                    if len(bg_dose) > 0:
                        val_bg_mean_sum += bg_dose.mean().item()

                    n_val += 1

                    del outputs, outputs_activated, outputs_gy, body_mask_hard
                    del ptv_mask, bladder_mask, rectum_mask, ring_mask_val
                    del bg_mask, inputs, targets, normalized_targets
                    del bowel_mask_val, femur_mask_val
                    torch.cuda.empty_cache()

            val_loss_avg = val_loss_sum / max(n_val, 1)
            avg_mse = val_mse_sum / max(n_val, 1)
            avg_d95 = val_d95_sum / max(n_val, 1)
            avg_bladder = val_bladder_mean_sum / max(n_val, 1)
            avg_rectum = val_rectum_mean_sum / max(n_val, 1)
            avg_ring = val_ring_mean_sum / max(n_val, 1)
            avg_dmax = val_dmax_sum / max(n_val, 1)
            avg_bg = val_bg_mean_sum / max(n_val, 1)

        epoch_summary = (
            f"Epoch {epoch + 1} Summary: Train={train_loss_avg:.4f} Val={val_loss_avg:.4f} (MSE={avg_mse:.4f})\n"
            f"          Metrics: PTV_D95={avg_d95:.2f}Gy Bladder={avg_bladder:.2f}Gy Rectum={avg_rectum:.2f}Gy\n"
            f"          Metrics: Ring={avg_ring:.2f}Gy Dmax={avg_dmax:.2f}Gy BG={avg_bg:.2f}Gy"
        )
        print(f"  --> {epoch_summary}")
        logger.info(epoch_summary)

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(model.state_dict(), "best_dose_model.pth")
                checkpoint_msg = f"Saved best model val_loss={best_val_loss:.4f}"
                print(f"  --> {checkpoint_msg}")
                logger.info(checkpoint_msg)

    completion_msg = f"Training complete. Best val loss: {best_val_loss:.4f}"
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
    PHYSICAL_MAX_GY = 70.0   

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
        ("best_dose_model.pth",  "validation_summary.csv"),
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

                ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

                ptv_dose     = outputs_gy[ptv_mask.bool()].cpu()
                bladder_dose = outputs_gy[bladder_mask.bool()].cpu()
                rectum_dose  = outputs_gy[rectum_mask.bool()].cpu()

                try:
                    label_path = val_files[idx]["dose_label"]
                    patient_id = os.path.basename(label_path).replace(".nii.gz", "")
                except (IndexError, KeyError):
                    patient_id = f"patient_{idx:03d}"

                row = {"Patient_ID": patient_id}

                for pct in (99, 98, 95, 50, 2):
                    row[f"PTV_D{pct} (Gy)"] = quantile_dose(ptv_dose, pct)
                row["PTV_Mean (Gy)"] = ptv_dose.mean().item() if ptv_dose.numel() else float("nan")
                row["PTV_Max (Gy)"]  = ptv_dose.max().item()  if ptv_dose.numel() else float("nan")

                row["Bladder_Mean (Gy)"] = bladder_dose.mean().item() if bladder_dose.numel() else float("nan")
                row["Bladder_Max (Gy)"]  = bladder_dose.max().item()  if bladder_dose.numel() else float("nan")
                for thresh in (60.4, 56.0, 47.0, 38.0, 28.6):
                    row[f"Bladder_V{thresh}Gy (%)"] = v_metric(bladder_dose, thresh)

                row["Rectum_Mean (Gy)"] = rectum_dose.mean().item() if rectum_dose.numel() else float("nan")
                row["Rectum_Max (Gy)"]  = rectum_dose.max().item()  if rectum_dose.numel() else float("nan")
                for thresh in (60.4, 56.0, 47.0, 38.0, 28.6):
                    row[f"Rectum_V{thresh}Gy (%)"] = v_metric(rectum_dose, thresh)

                records.append(row)
                print(
                    f"  [{idx + 1}/{len(val_loader)}] {patient_id}  "
                    f"PTV D95={row['PTV_D95 (Gy)']:.2f} Gy  "
                    f"Bladder Mean={row['Bladder_Mean (Gy)']:.2f} Gy  "
                    f"Rectum Mean={row['Rectum_Mean (Gy)']:.2f} Gy"
                )

        df = pd.DataFrame(records)
        float_cols = [c for c in df.columns if c != "Patient_ID"]
        df[float_cols] = df[float_cols].round(2)
        df.to_csv(csv_name, index=False)
        print(f"\n  Saved '{csv_name}'")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
