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
)
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference

# ===================================================================
# 1. Configuration
# ===================================================================
DATA_DIR = os.environ.get("DATA_DIR", "./nnUNet_raw/Dataset001_ProstateDose")
IMAGES_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR = os.path.join(DATA_DIR, "labelsTr")

CHANNELS = ["0000", "0001", "0002", "0003", "0004"]  # CT, PTV, Bladder SDM, Anorectum SDM, IMRT Beam Prior
TARGET_SPACING = (1.27, 1.27, 2.5)
PATCH_SIZE = (96, 96, 96)

PRESCRIPTION_DOSE_GY = 75.0  # Normalisation factor (Gy -> [0,1]); headroom above 66.34 Gy max Rx
CONSTRAINT_CSV = os.environ.get(
    "CONSTRAINT_CSV", "./prostate_prime_constraints_v2.csv"
)

# ===================================================================
# 2. Constraint Parsing  (Step 1)
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
    # Accumulate Optimal and Mandatory rows separately, then merge per dose level.
    # Key: (organ_canonical, dose_gy)  Value: {"optimal_v": ..., "mandatory_v": ...}
    v_accum = {}          # (organ, dose_gy) -> {"optimal_v": nan, "mandatory_v": nan}
    ptv_coverage = []
    ptv_max_dose_gy = None

    # Determine the Name suffix that signals patient class
    # N0  -> no suffix (exact names like "Bladder", "Anorectum", "PTV62" ...)
    # N+  -> suffix "_Nplus"
    nplus_suffix = "_Nplus"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name        = row["Name"].strip()
            struct_type = row["Type"].strip()           # PTV / CTV / Avoidance
            ctype       = row["Constraint_Type"].strip() # D or V
            c_val_raw   = row["Constraint_Value"].strip()
            c_unit      = row["Constraint_Unit"].strip()
            obj_val_raw = row["Objective_Value"].strip()
            obj_unit    = row["Objective_Unit"].strip()
            obj_type    = row["Objective_Type"].strip()  # Optimal / Mandatory

            # Skip rows with no objective value (empty rules)
            if not obj_val_raw:
                continue

            # ---- Determine patient class from Name ------------------
            is_nplus = name.endswith(nplus_suffix)
            if patient_class == "N0" and is_nplus:
                continue
            if patient_class == "N+" and not is_nplus:
                continue

            # Canonical organ name (strip the _Nplus suffix if present)
            canonical = name[: -len(nplus_suffix)] if is_nplus else name

            # ---- V-Type constraints (Bladder / Anorectum) -----------
            if ctype == "V" and canonical in ("Bladder", "Anorectum"):
                # Only fractional-volume limits (Objective_Unit == %)
                if obj_unit.strip() != "%":
                    continue  # skip cc-based rules (Small_Bowel etc.)
                # Constraint_Value is the dose threshold in Gy
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
            # Constraint_Value is the percentile (e.g. "95", "98") and
            # Constraint_Unit is "%".  Objective_Value is the required
            # dose as a fraction of prescription (e.g. 0.95).
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
                            "fraction": float(obj_val_raw),  # e.g. 0.95
                        }
                    )

    # ---- Build the final v_constraints list (per organ) -------------
    v_constraints = {"Bladder": [], "Anorectum": []}
    for (organ, dose_gy), tiers in sorted(v_accum.items(),
                                          key=lambda x: x[0][1]):
        # Only include rules that have at least a mandatory limit
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
# 3. Physics-Guided Loss Module  (Step 2)
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
        lambda_mse=10.0,        # Anchors the dose to the human ground truth
        lambda_optimal=2.0,     # A gentle nudge to keep OARs as low as possible
        lambda_mandatory=10.0,  # A strict brick wall to protect the organs
        lambda_ptv=10.0,        # Fiercely forces the model to hit the 57+ Gy target
        lambda_smooth=0.1,      # Prevents jagged isodose lines
        lambda_ring=15.0,       # Penalises hot-spots in the 5mm PTV falloff ring
        k_steepness=50.0,       # Perfect for [0, 1] normalized space
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
        self.k = k_steepness

    # --- Differentiable DVH volume fraction --------------------------
    def calculate_dvh_volume(self, predicted_dose, organ_mask, norm_dose_threshold):
        """
        V^pred_{D_ref} = (1/N_OAR) * Σ_{i∈OAR} σ(k·(D_i - D_ref))

        Uses torch.sigmoid — fully differentiable.
        organ_mask: binary float tensor, same spatial shape as predicted_dose.
        """
        # Flatten spatial dims for the masked organ
        organ_voxels = predicted_dose * organ_mask         # zero outside organ
        n_organ = organ_mask.sum()

        if n_organ < 1.0:
            return torch.tensor(0.0, device=predicted_dose.device,
                                dtype=predicted_dose.dtype)

        step_approx = torch.sigmoid(self.k * (organ_voxels - norm_dose_threshold))
        # Only count voxels inside the organ
        step_approx = step_approx * organ_mask
        volume_fraction = step_approx.sum() / n_organ
        return volume_fraction

    # --- Forward pass ------------------------------------------------
    def forward(self, pred_dose, true_dose, bladder_mask, rectum_mask,
                ptv_mask, ring_mask):
        """
        Parameters
        ----------
        pred_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        true_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        bladder_mask: (B, 1, D, H, W)  — binary float
        rectum_mask : (B, 1, D, H, W)  — binary float
        ptv_mask    : (B, 1, D, H, W)  — binary float
        ring_mask   : (B, 1, D, H, W)  — binary float, 5mm shell around PTV
        """
        # ------ 1. L_MSE  -------------------------------------------
        loss_mse = self.mse(pred_dose, true_dose)

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
                # Optimal tier
                if not math.isnan(rule["optimal_v"]):
                    viol_opt = torch.relu(v_frac - rule["optimal_v"])
                    loss_optional = loss_optional + viol_opt ** 2

                # Mandatory tier
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

        # ------ 4. L_D-Type max dose (PTV max) ----------------------
        max_gy = self.constraints["d_type"]["PTV_max_dose_gy"]
        if max_gy is not None:
            max_norm = max_gy / PRESCRIPTION_DOSE_GY
            if ptv_n > 0:
                overdose = torch.relu(pred_dose - max_norm) * ptv_mask
                loss_ptv = loss_ptv + (overdose ** 2).sum() / ptv_n

        # ------ 5. L_Ring (Falloff shell penalty) -------------------
        # Any ring voxel exceeding 55% of prescription (41.25/75 Gy = 0.55
        # in normalised space) is penalised with a squared hinge loss.
        # This prevents the EBRT model from simulating an isotropic
        # brachytherapy-like dose cloud that floods lateral healthy tissue.
        RING_THRESH = 0.55  # = 41.25 Gy / 75 Gy
        ring_n = ring_mask.sum()
        if ring_n > 0:
            ring_overdose = torch.relu(pred_dose - RING_THRESH) * ring_mask
            loss_ring = (ring_overdose ** 2).sum() / ring_n
        else:
            loss_ring = torch.tensor(0.0, device=pred_dose.device,
                                     dtype=pred_dose.dtype)

        # ------ 6. L_smooth (Total Variation) -----------------------
        gd = pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :]
        gh = pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :]
        gw = pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1]
        loss_smooth = (torch.mean(gd ** 2) + torch.mean(gh ** 2)
                       + torch.mean(gw ** 2))

        # ------ Total -----------------------------------------------
        total = (
            self.lambda_mse       * loss_mse
            + self.lambda_optimal * loss_optional
            + self.lambda_mandatory * loss_mandatory
            + self.lambda_ptv     * loss_ptv
            + self.lambda_ring    * loss_ring
            + self.lambda_smooth  * loss_smooth
        )
        return total, {
            "mse":    loss_mse.item(),
            "v_opt":  loss_optional.item(),
            "v_mand": loss_mandatory.item(),
            "ptv":    loss_ptv.item(),
            "ring":   loss_ring.item(),
            "smooth": loss_smooth.item(),
        }


# ===================================================================
# 4. Data Loading  (Step 3)
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
        # ch_0 contains raw HU values at this point (before NormalizeIntensityd)
        ct      = d["ch_0"]           # (1, D, H, W)
        ptv     = d["ch_1"] >= 0.5    # PTV binary mask
        bladder = d["ch_2"] <= 0.0    # Bladder SDM: inside = <= 0
        rectum  = d["ch_3"] <= 0.0    # Rectum  SDM: inside = <= 0
        body    = ct >= self.body_thresh_hu   # Patient body vs air

        # Start with Class 0 (Air) everywhere
        crop_mask = torch.zeros_like(ct, dtype=torch.float32)
        # Layer up in priority order (later writes win)
        crop_mask[body]    = 1.0   # Healthy tissue
        crop_mask[ptv]     = 2.0   # PTV
        crop_mask[bladder] = 3.0   # Bladder
        crop_mask[rectum]  = 4.0   # Rectum

        d["crop_mask"] = crop_mask
        return d


# ===================================================================
# 4b. Falloff Ring: morphological dilation of PTV by ~5 mm
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
    needs_batch = ptv_binary.ndim == 4          # (1,D,H,W) — add batch dim
    if needs_batch:
        ptv_binary = ptv_binary.unsqueeze(0)    # → (1,1,D,H,W)

    dilated = F.max_pool3d(
        ptv_binary.float(),
        kernel_size=(5, 9, 9),
        stride=1,
        padding=(2, 4, 4),
    )
    ring = torch.clamp(dilated - ptv_binary.float(), min=0.0)

    if needs_batch:
        ring = ring.squeeze(0)                  # back to (1,D,H,W)
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
        ptv = (d["ch_1"] >= 0.5).float()   # (1, D, H, W) on CPU in transforms
        d["ring_mask"] = compute_ring_mask(ptv)
        return d


ALL_KEYS = ["ch_0", "ch_1", "ch_2", "ch_3", "ch_4", "dose_label"]

train_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear", "bilinear"),
        ),
        # ── Build 5-class crop mask BEFORE CT normalisation ──────────────
        # ch_0 must still contain raw HU values here so the body/air
        # threshold (-300 HU) correctly separates patient tissue from air.
        # After this transform, NormalizeIntensityd z-scores ch_0.
        Create5ClassCropMaskd(keys=["ch_0"], body_thresh_hu=-300.0),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        # ── Falloff ring mask (5 mm PTV shell) ───────────────────────────
        # Runs after Spacingd (ch_1 at TARGET_SPACING) and before cropping
        # so ring_mask is cropped identically to the image patches.
        # Uses F.max_pool3d — pure PyTorch, no CPU/GPU round-trip.
        CreateFalloffRingd(keys=["ch_1"]),
        # ── Balanced 5-class patch sampling ──────────────────────────────
        # ratios=[0,1,1,1,1] → Air never sampled; 25% per active class.
        # num_samples=4 → exactly 1 patch per active class per patient,
        # preventing the model from ignoring healthy tissue (the "brachytherapy
        # loophole" where unpenalised dose floods lateral hip tissue).
        # If CUDA OOM occurs, reduce num_samples to 2 (stochastic 50/50).
        RandCropByLabelClassesd(
            keys=ALL_KEYS + ["ring_mask"],   # ring_mask must be cropped too
            label_key="crop_mask",
            spatial_size=PATCH_SIZE,
            num_classes=5,
            ratios=[0.0, 1.0, 1.0, 1.0, 1.0],
            num_samples=4,
        ),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "ch_4"], name="image"),
        ToTensord(keys=["image", "dose_label", "ring_mask"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear", "bilinear"),
        ),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3", "ch_4"], name="image"),
        ToTensord(keys=["image", "dose_label"]),
    ]
)


# ===================================================================
# 5. Helper: extract binary masks from the 4-channel input tensor
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
# 6. Main Training Loop  (Step 4)
# ===================================================================

def main():
    # ---- Load constraints ------------------------------------------
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

    # 80/20 split by default; override with VAL_SPLIT env var (e.g. 0.15)
    val_frac   = float(os.environ.get("VAL_SPLIT", "0.20"))
    n_val      = max(1, round(n_total * val_frac))
    n_train    = n_total - n_val
    train_files = data_dicts[:n_train]
    val_files   = data_dicts[n_train:]
    print(f"Split: {n_train} train  /  {n_val} val  (val_frac={val_frac:.0%})")

    # ---- Batch size and workers (env-var overrideable) ---------------
    # BATCH_SIZE=2 doubles VRAM usage (each sample = 4 crops of 96³).
    # For a 24 GB GPU, BATCH_SIZE=1 is safe; BATCH_SIZE=2 may work.
    batch_size   = int(os.environ.get("BATCH_SIZE", "1"))
    num_workers  = min(os.cpu_count() or 4, 8)
    print(f"Batch size={batch_size}  num_workers={num_workers}")

    cache_dir = os.path.join(DATA_DIR, "persistent_cache_physics")
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,
    )

    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,              # always 1 for val — sliding window handles the patch
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model -----------------------------------------------------
    print("Building 3D U-Net...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # ---- Loss / Optimizer / Scheduler ------------------------------
    loss_function = PhysicsGuidedDoseLoss(
        constraints_dict=constraints,
        lambda_mse=10.0,
        lambda_optimal=2.0,
        lambda_mandatory=25.0,
        lambda_ptv=10.0,
        lambda_ring=15.0,       # falloff shell penalty
        lambda_smooth=0.1,
        k_steepness=50.0,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print(f"Model and dataloaders ready on {device}!")
    print(f"Epochs={epochs}  Scheduler=CosineAnnealingLR  "
          f"AMP={'ON' if torch.cuda.is_available() else 'OFF'}\n")

    # ---- Training --------------------------------------------------
    best_val_loss      = float("inf")   # tracks physics (loss) optimum
    best_clinical_score = float("inf")  # tracks clinical (soft-margin) optimum

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}  "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

        # ---- Train -------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            inputs  = batch["image"].to(device)
            targets = batch["dose_label"].to(device)
            # ring_mask is pre-cropped to the patch size by RandCropByLabelClassesd
            ring_mask_batch = batch["ring_mask"].to(device)

            # Normalise dose targets to [0, 1]
            normalized_targets = targets / PRESCRIPTION_DOSE_GY

            # SDM -> binary masks
            ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
                dtype=torch.float16,
            ):
                outputs = model(inputs)

            # Apply Softplus OUTSIDE autocast in float32.
            # Reason: float16 saturates at ~65504 — applying Softplus inside AMP
            # could silently overflow and reproduce the 350 Gy explosion.
            # Softplus ensures all voxel predictions are physically positive (> 0)
            # while maintaining full gradient flow, unlike Sigmoid which saturates.
            outputs_activated = F.softplus(outputs.float())

            # Physics loss receives the Softplus-activated, float32 output
            loss, components = loss_function(
                outputs_activated,
                normalized_targets.float(),
                bladder_mask.float(),
                rectum_mask.float(),
                ptv_mask.float(),
                ring_mask_batch.float(),
            )

            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping for mixed precision
            scaler.unscale_(optimizer)
            # Clip gradients to prevent exploding gradients from custom physics loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            if step % 5 == 0 or step == len(train_loader):
                print(
                    f"  Step {step}/{len(train_loader)}  "
                    f"Loss={loss.item():.4f}  "
                    f"[mse={components['mse']:.4f}  "
                    f"v_opt={components['v_opt']:.5f}  "
                    f"v_mand={components['v_mand']:.5f}  "
                    f"ptv={components['ptv']:.4f}  "
                    f"ring={components['ring']:.5f}  "
                    f"smooth={components['smooth']:.4f}]"
                )

        train_loss_avg = train_loss_sum / max(len(train_loader), 1)

        # Step scheduler AFTER each epoch
        scheduler.step()

        # ---- Validation --------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_d95_sum = 0.0
        val_bladder_mean_sum = 0.0
        val_rectum_mean_sum = 0.0
        n_val = 0

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

                # Softplus in float32, outside AMP (same reason as training loop)
                outputs_activated = F.softplus(outputs.float())

                normalized_targets = targets / PRESCRIPTION_DOSE_GY
                ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

                # Ring mask computed on-the-fly for validation.
                # val_transforms has no cropping, so the full-volume ptv_mask
                # already matches the full-volume sliding-window outputs.
                ring_mask_val = compute_ring_mask(ptv_mask)

                loss, _ = loss_function(
                    outputs_activated,
                    normalized_targets.float(),
                    bladder_mask.float(),
                    rectum_mask.float(),
                    ptv_mask.float(),
                    ring_mask_val.float(),
                )
                val_loss_sum += loss.item()

                # Clinical metrics — denormalise the Softplus-activated output
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

                n_val += 1

        val_loss_avg = val_loss_sum / max(n_val, 1)
        avg_d95 = val_d95_sum / max(n_val, 1)
        avg_bladder = val_bladder_mean_sum / max(n_val, 1)
        avg_rectum = val_rectum_mean_sum / max(n_val, 1)

        print(
            f"  --> Epoch {epoch + 1} Summary:  "
            f"Train={train_loss_avg:.4f}  Val={val_loss_avg:.4f}"
        )
        print(
            f"      Clinical:  PTV D95={avg_d95:.2f} Gy  "
            f"Bladder Mean={avg_bladder:.2f} Gy  "
            f"Rectum Mean={avg_rectum:.2f} Gy"
        )

        # -- Physics checkpoint: best validation loss --------------------
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_dose_model_physics.pth")
            print(f"  --> [PHYSICS]  Saved best PHYSICS model  "
                  f"(val_loss={best_val_loss:.4f})")

        # -- Clinical checkpoint: soft-margin exchange-rate score --------
        # Penalises OAR toxicity AND PTV underdose simultaneously.
        # Exchange rate: 1 Gy PTV deficit = 3 Gy penalty.
        current_worst_oar = max(avg_bladder, avg_rectum)
        ptv_deficit       = max(0.0, 62.4 - avg_d95)      # deficit below 62.4 Gy ideal
        clinical_score    = current_worst_oar + (ptv_deficit * 3.0)

        if clinical_score < best_clinical_score:
            best_clinical_score = clinical_score
            torch.save(model.state_dict(), "best_dose_model_clinical.pth")
            print(f"  --> [CLINICAL] Saved best CLINICAL model  "
                  f"Score={clinical_score:.3f}  "
                  f"PTV_D95={avg_d95:.2f} Gy  "
                  f"Bladder={avg_bladder:.2f} Gy  "
                  f"Rectum={avg_rectum:.2f} Gy")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}  "
          f"Best clinical score: {best_clinical_score:.4f}")

    # ====================================================================
    # FINAL CLINICAL EVALUATION BLOCK
    # Iterates over both saved checkpoints (Physics + Clinical) and
    # produces a separate patient-by-patient DVH summary CSV for each.
    # ====================================================================
    print("\n" + "=" * 68)
    print("FINAL CLINICAL EVALUATION  (Physics + Clinical checkpoints)")
    print("=" * 68)

    # ---- Imports needed only for this block ----------------------------
    import pandas as pd  # pyrefly: ignore [missing-import]

    # ---- Physical dose ceiling (Gy) — hard clinical constraint ---------
    PHYSICAL_MAX_GY = 70.0   # no voxel can receive more than this

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
        ("best_dose_model_physics.pth",  "validation_physics_summary.csv"),
        ("best_dose_model_clinical.pth", "validation_clinical_summary.csv"),
    ]:
        print(f"\n--- Evaluating: {model_path} -> {csv_name} ---")

        # Load checkpoint
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

                # Sliding-window inference in AMP
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

                # Softplus in float32 outside AMP, then denormalise → clamp.
                # 70.0 Gy clamp = clinical prescription ceiling (NOT the 75 Gy
                # normalisation constant). Softplus handles training stability;
                # clamp handles clinical reporting safety.
                outputs_gy = F.softplus(outputs.float()) * PRESCRIPTION_DOSE_GY
                outputs_gy = torch.clamp(outputs_gy, min=0.0, max=PHYSICAL_MAX_GY)

                # Binary masks
                ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

                ptv_dose     = outputs_gy[ptv_mask.bool()].cpu()
                bladder_dose = outputs_gy[bladder_mask.bool()].cpu()
                rectum_dose  = outputs_gy[rectum_mask.bool()].cpu()

                # Patient ID
                try:
                    label_path = val_files[idx]["dose_label"]
                    patient_id = os.path.basename(label_path).replace(".nii.gz", "")
                except (IndexError, KeyError):
                    patient_id = f"patient_{idx:03d}"

                # Metrics
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

        # Export CSV
        df = pd.DataFrame(records)
        float_cols = [c for c in df.columns if c != "Patient_ID"]
        df[float_cols] = df[float_cols].round(2)
        df.to_csv(csv_name, index=False)
        print(f"\n  Saved '{csv_name}'")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
