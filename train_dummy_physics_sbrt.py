"""
train_dummy_physics_sbrt.py
============================
Physics-Guided SBRT Prostate Dose Prediction.
Fork of train_dummy_physics_new.py — key differences:
  - Dynamic rx_gy and n_status per patient (from sbrt_manifest.csv)
  - No SIB: single PTV per patient
  - No Beam Prior / Body Mask / Penile Bulb channels
  - Channels: CT | PTV | Bladder_SDM | Anorectum_SDM | Small_Bowel | Femur_R | Femur_L
  - PRIME v3.1 SBRT OAR constraints (N0 vs N+ conditional)
  - Normalization constant = 42.0 Gy (>= 115% of max Rx 36.25 Gy)
"""

import os, glob, math, csv, logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from monai.data import PersistentDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, ToTensord, ConcatItemsd, MapTransform,
    RandCropByLabelClassesd, DeleteItemsd,
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# ===================================================================
# CONFIGURATION
# ===================================================================
DATA_DIR        = os.environ.get("DATA_DIR", "./nnUNet_raw/Dataset002_SBRTProstate")
IMAGES_DIR      = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR      = os.path.join(DATA_DIR, "labelsTr")
MANIFEST_PATH   = os.path.join(DATA_DIR, "sbrt_manifest.csv")

# 7 channels: 0=CT 1=PTV 2=Bladder_SDM 3=Anorectum_SDM 4=Small_Bowel 5=Femur_R 6=Femur_L
CHANNELS        = ["0000", "0001", "0002", "0003", "0004", "0005", "0006"]
TARGET_SPACING  = (1.27, 1.27, 2.5)
PATCH_SIZE      = (128, 128, 64)

# Normalization constant — must be >= 115% of highest possible Rx (36.25 * 1.15 = 41.7)
NORM_GY         = 42.0

GRAD_ACCUM_STEPS    = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
VAL_EVERY_N_EPOCHS  = int(os.environ.get("VAL_EVERY_N_EPOCHS", "1"))
WARMUP_EPOCHS       = int(os.environ.get("WARMUP_EPOCHS", "75"))

# ===================================================================
# SBRT PHYSICS-GUIDED LOSS ENGINE
# ===================================================================
class SBRTPhysicsLoss(nn.Module):
    """
    Bucket loss for SBRT:
      - PTV floor  = rx_gy / NORM_GY
      - PTV ceiling = rx_gy * 1.07 / NORM_GY
    OAR V-constraints are dynamically selected by n_status (0=N0, 1=N+).
    Small Bowel uses absolute volume (cc) thresholds.
    """

    # PRIME v3.1 SBRT OAR constraint tables
    # Format: (dose_gy, max_vol_fraction)
    BLADDER_N0  = [(35.0, 0.04), (31.5, 0.08), (28.0, 0.10), (17.5, 0.20), (14.0, 0.35)]
    BLADDER_NP  = [(35.0, 0.04), (31.5, 0.08), (28.0, 0.12), (17.5, 0.28), (14.0, 0.35)]
    ANORECT_N0  = [(35.0, 0.05), (31.5, 0.10), (28.0, 0.15), (17.5, 0.35), (14.0, 0.45)]
    ANORECT_NP  = [(35.0, 0.05), (31.5, 0.10), (28.0, 0.15), (17.5, 0.35), (14.0, 0.50)]
    # Small Bowel: absolute volume (cc)
    SMALL_BOWEL_CC = [(28.0, 80.0), (27.5, 2.0)]
    # Femoral heads: volume fraction
    FEMUR_RULES = [(14.0, 0.05)]

    def __init__(
        self,
        lambda_mse=25.0,
        lambda_ptv=0.0,
        lambda_ptv_max=0.0,
        lambda_oar=0.0,
        lambda_smooth=0.0,
        lambda_anticollapse=0.0,
        lambda_ring=0.0,
        k_steepness=50.0,
    ):
        super().__init__()
        self.lambda_mse          = lambda_mse
        self.lambda_ptv          = lambda_ptv
        self.lambda_ptv_max      = lambda_ptv_max
        self.lambda_oar          = lambda_oar
        self.lambda_smooth       = lambda_smooth
        self.lambda_anticollapse = lambda_anticollapse
        self.lambda_ring         = lambda_ring
        self.k                   = k_steepness

    def _dvh_vfrac(self, pred, mask, norm_thresh):
        """Differentiable DVH volume fraction via sigmoid approximation."""
        n = mask.sum()
        if n < 1.0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        sig = torch.sigmoid(self.k * (pred * mask - norm_thresh)) * mask
        return sig.sum() / n

    def forward(self, pred_dose, true_dose, inputs, ring_mask, rx_gy, n_status):
        """
        pred_dose  : (B,1,D,H,W) normalised [0,1]
        true_dose  : (B,1,D,H,W) normalised [0,1]
        inputs     : (B,7,D,H,W)
        ring_mask  : (B,1,D,H,W)
        rx_gy      : float — patient Rx dose (36.25, 30, or 25 Gy)
        n_status   : int  — 0=N0, 1=N+
        """
        # Extract masks from channel layout
        ptv_mask        = (inputs[:, 1:2, ...] >= 0.5).float()
        bladder_mask    = (inputs[:, 2:3, ...] <= 0.0).float()
        anorectum_mask  = (inputs[:, 3:4, ...] <= 0.0).float()
        small_bowel_mask = (inputs[:, 4:5, ...] > 0.5).float()
        femur_r_mask    = (inputs[:, 5:6, ...] > 0.5).float()
        femur_l_mask    = (inputs[:, 6:7, ...] > 0.5).float()

        # ------ 1. MSE (dose-weighted) ----------------------------------
        DOSE_WEIGHT_SCALE = 9.0
        dose_weight = 1.0 + DOSE_WEIGHT_SCALE * true_dose.clamp(max=0.96)
        loss_mse = ((pred_dose - true_dose) ** 2 * dose_weight).mean()

        # ------ 2. PTV Bucket Loss (hinge floor + ceiling) --------------
        rx_norm   = rx_gy / NORM_GY
        ceil_norm = (rx_gy * 1.07) / NORM_GY

        ptv_n = ptv_mask.sum().clamp(min=1.0)
        underdose  = (torch.relu(rx_norm   - pred_dose) ** 2) * ptv_mask
        overdose   = (torch.relu(pred_dose - ceil_norm) ** 2) * ptv_mask
        loss_ptv     = underdose.sum() / ptv_n
        loss_ptv_max = overdose.sum()  / ptv_n

        # ------ 3. Anti-Collapse safety net -----------------------------
        loss_anticollapse = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        if ptv_n > 1.0:
            frac = (torch.relu(0.50 - pred_dose) * ptv_mask).sum() / ptv_n
            loss_anticollapse = frac ** 2

        # ------ 4. OAR V-constraints (dynamic N0 / N+) -----------------
        bladder_rules  = self.BLADDER_NP  if n_status else self.BLADDER_N0
        anorect_rules  = self.ANORECT_NP  if n_status else self.ANORECT_N0

        loss_oar = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)

        for dose_gy, max_vfrac in bladder_rules:
            norm_thresh = dose_gy / rx_gy  # normalise by Rx dose per patient
            vf = self._dvh_vfrac(pred_dose, bladder_mask, norm_thresh)
            loss_oar = loss_oar + torch.relu(vf - max_vfrac) ** 2

        for dose_gy, max_vfrac in anorect_rules:
            norm_thresh = dose_gy / rx_gy
            vf = self._dvh_vfrac(pred_dose, anorectum_mask, norm_thresh)
            loss_oar = loss_oar + torch.relu(vf - max_vfrac) ** 2

        # Small Bowel — absolute volume (cc).
        # Voxel volume in cc from pred_dose spatial shape is approximated
        # from TARGET_SPACING (1.27 x 1.27 x 2.5 mm) = 4.031 mm³ = 0.004031 cc
        VOXEL_CC = (1.27 * 1.27 * 2.5) / 1000.0  # ≈ 0.004031 cc
        sb_n = small_bowel_mask.sum().clamp(min=1.0)
        if sb_n > 1.0:
            for dose_gy, max_cc in self.SMALL_BOWEL_CC:
                norm_thresh = dose_gy / rx_gy
                sig = torch.sigmoid(self.k * (pred_dose * small_bowel_mask - norm_thresh))
                vol_cc = (sig * small_bowel_mask).sum() * VOXEL_CC
                loss_oar = loss_oar + torch.relu(vol_cc - max_cc) ** 2

        # Femoral Heads — V14Gy < 5%
        for fmask in [femur_r_mask, femur_l_mask]:
            fn = fmask.sum()
            if fn > 0:
                norm_thresh = 14.0 / rx_gy
                vf = self._dvh_vfrac(pred_dose, fmask, norm_thresh)
                loss_oar = loss_oar + torch.relu(vf - 0.05) ** 2

        # ------ 5. Ring falloff penalty --------------------------------
        loss_ring = torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
        ring_n = ring_mask.sum()
        if ring_n > 0:
            ring_thresh = rx_norm * 0.88
            ring_over = torch.relu(pred_dose - ring_thresh) * ring_mask
            loss_ring = (ring_over ** 2).sum() / ring_n

        # ------ 6. Spatial smoothness ----------------------------------
        def _tv(t):
            return (
                (t[:, :, 1:, :, :] - t[:, :, :-1, :, :]).abs().mean() +
                (t[:, :, :, 1:, :] - t[:, :, :, :-1, :]).abs().mean() +
                (t[:, :, :, :, 1:] - t[:, :, :, :, :-1]).abs().mean()
            )
        loss_smooth = _tv(pred_dose)

        # ------ Total --------------------------------------------------
        total = (
            self.lambda_mse          * loss_mse
            + self.lambda_ptv        * loss_ptv
            + self.lambda_ptv_max    * loss_ptv_max
            + self.lambda_oar        * loss_oar
            + self.lambda_smooth     * loss_smooth
            + self.lambda_anticollapse * loss_anticollapse
            + self.lambda_ring       * loss_ring
        )
        return total, {
            "mse":          loss_mse.item(),
            "ptv":          loss_ptv.item(),
            "ptv_max":      loss_ptv_max.item(),
            "oar":          loss_oar.item(),
            "smooth":       loss_smooth.item(),
            "anticollapse": loss_anticollapse.item(),
            "ring":         loss_ring.item(),
        }


# ===================================================================
# DATA LOADING
# ===================================================================
def get_data_dicts():
    """
    Build data dicts from labelsTr. Injects rx_gy and n_status
    from sbrt_manifest.csv into each patient dict.
    """
    # Load manifest
    manifest = {}
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[row["patient_id"].strip()] = {
                    "rx_gy":    float(row["rx_gy"]),
                    "n_status": int(row["n_status"]),
                }
    else:
        print(f"WARNING: {MANIFEST_PATH} not found — defaulting rx_gy=36.25, n_status=0 for all patients")

    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.nii.gz")))
    data_dicts = []
    for label_path in label_files:
        patient_id = os.path.basename(label_path).replace(".nii.gz", "")
        pt_dict = {"dose_label": label_path, "patient_id": patient_id}
        for i, ch in enumerate(CHANNELS):
            pt_dict[f"ch_{i}"] = os.path.join(IMAGES_DIR, f"{patient_id}_{ch}.nii.gz")

        # Inject metadata — NOT added to ALL_KEYS (bypasses spatial transforms)
        meta = manifest.get(patient_id, {"rx_gy": 36.25, "n_status": 0})
        pt_dict["rx_gy"]    = meta["rx_gy"]
        pt_dict["n_status"] = meta["n_status"]

        data_dicts.append(pt_dict)
    return data_dicts


# ===================================================================
# TRANSFORMS
# ===================================================================
class CreateSBRTPTVMapd(MapTransform):
    """
    SBRT has a single PTV. Channel 1 is already binary [0,1].
    This pass-through transform simply renames ch_1 → discrete_ptv
    to keep the pipeline consistent with the IMRT fork.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        d["discrete_ptv"] = (d["ch_1"] >= 0.5).float()
        return d


class CreateCropMaskd(MapTransform):
    """3-class crop mask: background | PTV | body."""
    def __init__(self, keys, body_thresh_hu=-300.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.thresh = body_thresh_hu

    def __call__(self, data):
        d = dict(data)
        ct = d["ch_0"]
        ptv = (d["ch_1"] >= 0.5)
        body = (ct > self.thresh / 1000.0)  # normalised later; raw HU here
        crop = torch.zeros_like(ct, dtype=torch.long)
        crop[body] = 1
        crop[ptv]  = 2
        d["crop_mask"] = crop.float()
        return d


class CreateRingMaskd(MapTransform):
    """5 mm falloff ring around PTV via max-pool dilation."""
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        ptv = (d["ch_1"] >= 0.5).float()
        dilated = F.max_pool3d(ptv.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        d["ring_mask"] = (dilated - ptv).clamp(0, 1)
        return d


# Spatial keys that go through LoadImaged + Spacingd
CH_KEYS  = [f"ch_{i}" for i in range(7)]
ALL_KEYS = CH_KEYS + ["dose_label"]

train_transforms = Compose([
    LoadImaged(keys=ALL_KEYS, allow_missing_keys=True),
    EnsureChannelFirstd(keys=ALL_KEYS, allow_missing_keys=True),
    Spacingd(
        keys=ALL_KEYS,
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear",
              "nearest", "nearest", "nearest",   # ch_0 … ch_6
              "bilinear"),                        # dose_label
        allow_missing_keys=True,
    ),
    CreateSBRTPTVMapd(keys=["ch_0"]),
    CreateCropMaskd(keys=["ch_0"], body_thresh_hu=-300.0),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    CreateRingMaskd(keys=["ch_0"]),
    RandCropByLabelClassesd(
        keys=ALL_KEYS + ["ring_mask", "discrete_ptv"],
        label_key="crop_mask",
        spatial_size=PATCH_SIZE,
        num_classes=3,
        ratios=[0.0, 1.0, 2.0],
        num_samples=2,
    ),
    DeleteItemsd(keys=["crop_mask"]),
    ConcatItemsd(
        keys=["ch_0", "discrete_ptv", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"],
        name="image",
    ),
    ToTensord(keys=["image", "dose_label", "ring_mask"]),
])

val_transforms = Compose([
    LoadImaged(keys=ALL_KEYS, allow_missing_keys=True),
    EnsureChannelFirstd(keys=ALL_KEYS, allow_missing_keys=True),
    Spacingd(
        keys=ALL_KEYS,
        pixdim=TARGET_SPACING,
        mode=("bilinear", "nearest", "bilinear", "bilinear",
              "nearest", "nearest", "nearest",
              "bilinear"),
        allow_missing_keys=True,
    ),
    CreateSBRTPTVMapd(keys=["ch_0"]),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(
        keys=["ch_0", "discrete_ptv", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6"],
        name="image",
    ),
    ToTensord(keys=["image", "dose_label"]),
])


def compute_ring_mask(ptv_mask):
    dilated = F.max_pool3d(ptv_mask, kernel_size=5, stride=1, padding=2)
    return (dilated - ptv_mask).clamp(0, 1)


# ===================================================================
# MAIN
# ===================================================================
def main():
    os.makedirs("logs", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/sbrt_training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"SBRT training started — log: {log_filename}")
    logger.info(f"NORM_GY={NORM_GY}, PATCH_SIZE={PATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM_STEPS}")

    # ---- Data ----------------------------------------------------------
    data_dicts = get_data_dicts()
    n_total = len(data_dicts)
    print(f"Found {n_total} SBRT patients.")

    val_frac  = float(os.environ.get("VAL_SPLIT", "0.20"))
    n_val_    = max(1, round(n_total * val_frac))
    n_train_  = n_total - n_val_
    train_files = data_dicts[:n_train_]
    val_files   = data_dicts[n_train_:]
    print(f"Split: {n_train_} train / {n_val_} val")

    cache_dir = os.path.join(DATA_DIR, "persistent_cache_sbrt")
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=2, prefetch_factor=1,
        persistent_workers=False, pin_memory=False,
        collate_fn=list_data_collate, drop_last=True,
    )
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, prefetch_factor=1,
        persistent_workers=False, pin_memory=False,
    )

    # ---- Model ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UNet(
        spatial_dims=3,
        in_channels=7,   # CT | PTV | Bladder_SDM | Anorectum_SDM | SmallBowel | FemurR | FemurL
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # ---- Loss / Optimizer / Scheduler ----------------------------------
    loss_fn = SBRTPhysicsLoss(
        lambda_mse=25.0, lambda_ptv=0.0, lambda_ptv_max=0.0,
        lambda_oar=0.0,  lambda_smooth=0.0, lambda_anticollapse=0.0,
        lambda_ring=0.0, k_steepness=50.0,
    )

    PHYSICS_TARGETS = {
        "lambda_ptv":          30.0,
        "lambda_ptv_max":      30.0,
        "lambda_oar":          50.0,
        "lambda_smooth":        0.5,   # relaxed: allow steep OAR-PTV dose transitions
        "lambda_anticollapse": 150.0,
        "lambda_ring":         15.0,
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs    = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss       = float("inf")
    best_clinical_score = float("inf")
    best_mse            = float("inf")

    # ---- Training loop -------------------------------------------------
    for epoch in range(epochs):
        ramp = min((epoch + 1) / WARMUP_EPOCHS, 1.0)
        for attr, target in PHYSICS_TARGETS.items():
            setattr(loss_fn, attr, target * ramp)

        current_lr = optimizer.param_groups[0]["lr"]
        phase_tag  = f"RAMP {ramp:.0%}" if epoch < WARMUP_EPOCHS else "FULL PHYSICS"
        print(f"\nEpoch {epoch+1}/{epochs} [{phase_tag}] lr={current_lr:.2e}")

        model.train()
        train_loss_sum = 0.0
        accum = 0

        for batch in train_loader:
            accum += 1
            inputs  = batch["image"].to(device, non_blocking=True)
            targets = batch["dose_label"].to(device, non_blocking=True)
            ring_m  = batch["ring_mask"].to(device, non_blocking=True)

            # rx_gy and n_status come directly from the data dict (not spatially transformed)
            rx_gy    = float(batch["rx_gy"][0])
            n_status = int(batch["n_status"][0])

            norm_targets = targets / NORM_GY

            if accum == 1:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = model(inputs)

            outputs_act = F.softplus(outputs.float())

            loss, comps = loss_fn(outputs_act, norm_targets.float(), inputs, ring_m, rx_gy, n_status)
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

            if accum % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                accum = 0

            train_loss_sum += loss.item()

        # flush remaining grads
        if accum % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss_avg = train_loss_sum / max(len(train_loader), 1)
        scheduler.step()

        # ---- Validation ------------------------------------------------
        val_loss_avg = float("nan")
        avg_d95 = avg_d98 = avg_bladder = avg_rectum = avg_dmax = avg_ptv_deficit = float("nan")

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            model.eval()
            val_sum = d95_sum = d98_sum = bl_sum = re_sum = dmax_sum = 0.0
            ptv_deficit_sum = 0.0   # accumulated per-patient deficit against their own Rx
            n_v = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs  = batch["image"].to(device)
                    targets = batch["dose_label"].to(device)
                    rx_gy   = float(batch["rx_gy"][0])

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                        outputs = sliding_window_inference(
                            inputs=inputs, roi_size=PATCH_SIZE,
                            sw_batch_size=1, predictor=model, overlap=0.25,
                        )
                    outputs_act  = F.softplus(outputs.float())
                    norm_targets = targets / NORM_GY
                    mse = ((outputs_act - norm_targets.float()) ** 2).mean()
                    val_sum += mse.item()

                    outputs_gy = (outputs_act * NORM_GY).cpu()
                    ptv_mask_c  = (inputs[:, 1:2, ...] >= 0.5).cpu()
                    bl_mask_c   = (inputs[:, 2:3, ...] <= 0.0).cpu()
                    re_mask_c   = (inputs[:, 3:4, ...] <= 0.0).cpu()

                    ptv_dose = outputs_gy[ptv_mask_c.bool()]
                    if ptv_dose.numel() > 0:
                        patient_d95 = torch.quantile(ptv_dose, 0.05).item()
                        patient_d98 = torch.quantile(ptv_dose, 0.02).item()
                        d95_sum += patient_d95
                        d98_sum += patient_d98
                        # Per-patient deficit: how far D95 falls below THIS patient's own Rx
                        ptv_deficit_sum += max(0.0, rx_gy - patient_d95)

                    bl_dose = outputs_gy[bl_mask_c.bool()]
                    if bl_dose.numel() > 0: bl_sum += bl_dose.mean().item()
                    re_dose = outputs_gy[re_mask_c.bool()]
                    if re_dose.numel() > 0: re_sum += re_dose.mean().item()
                    dmax_sum += outputs_gy.max().item()
                    n_v += 1

                    del outputs_gy, ptv_dose, bl_dose, re_dose
                    del outputs, outputs_act, inputs, targets, norm_targets
                    del ptv_mask_c, bl_mask_c, re_mask_c
                    torch.cuda.empty_cache()

            if n_v:
                val_loss_avg   = val_sum     / n_v
                avg_d95        = d95_sum     / n_v
                avg_d98        = d98_sum     / n_v
                avg_bladder    = bl_sum      / n_v
                avg_rectum     = re_sum      / n_v
                avg_dmax       = dmax_sum    / n_v
                avg_ptv_deficit = ptv_deficit_sum / n_v  # mean Gy below Rx across cohort

        summary = (
            f"Epoch {epoch+1}: Train={train_loss_avg:.4f} Val={val_loss_avg:.4f}  "
            f"PTV D95={avg_d95:.2f}Gy D98={avg_d98:.2f}Gy  "
            f"Bladder={avg_bladder:.2f}Gy Rectum={avg_rectum:.2f}Gy Dmax={avg_dmax:.2f}Gy"
        )
        print(f"  --> {summary}")
        logger.info(summary)

        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
            is_valid = avg_dmax < (max(36.25, 30.0) * 1.15 * 2.0)  # sanity gate

            if is_valid and val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(model.state_dict(), "best_sbrt_model_physics.pth")
                logger.info(f"[PHYSICS] Saved best_sbrt_model_physics.pth  val={best_val_loss:.4f}")

            oar_score = max(avg_bladder, avg_rectum)
            # avg_ptv_deficit is already the mean per-patient Gy shortfall against each patient's own Rx
            clinical_score = oar_score + avg_ptv_deficit * 3.0
            if is_valid and clinical_score < best_clinical_score:
                best_clinical_score = clinical_score
                torch.save(model.state_dict(), "best_sbrt_model_clinical.pth")
                logger.info(
                    f"[CLINICAL] Saved best_sbrt_model_clinical.pth  score={clinical_score:.3f}  "
                    f"Deficit={avg_ptv_deficit:.2f}Gy PTV_D95={avg_d95:.2f}Gy PTV_D98={avg_d98:.2f}Gy  "
                    f"Bladder={avg_bladder:.2f}Gy Rectum={avg_rectum:.2f}Gy"
                )

            if val_loss_avg < best_mse:
                best_mse = val_loss_avg
                torch.save(model.state_dict(), "best_sbrt_model_diagnostic.pth")
                logger.info(f"[DIAGNOSTIC] Saved best_sbrt_model_diagnostic.pth  mse={best_mse:.4f}")

    logger.info(f"Training complete. best_val={best_val_loss:.4f} best_clinical={best_clinical_score:.4f}")

    # ====================================================================
    # FINAL CLINICAL EVALUATION
    # ====================================================================
    print("\n" + "="*68)
    print("SBRT FINAL CLINICAL EVALUATION")
    print("="*68)

    PHYSICAL_MAX_GY = 45.0

    def quantile_dose(d1d, pct):
        if d1d.numel() == 0: return float("nan")
        return torch.quantile(d1d.float(), 1.0 - pct / 100.0).item()

    def v_pct(d1d, thr):
        if d1d.numel() == 0: return float("nan")
        return ((d1d > thr).float().mean() * 100.0).item()

    def v_cc(d1d, thr):
        """Volume fraction > thr converted to cc using TARGET_SPACING."""
        VOXEL_CC = (1.27 * 1.27 * 2.5) / 1000.0
        if d1d.numel() == 0: return float("nan")
        n_over = (d1d > thr).float().sum().item()
        return n_over * VOXEL_CC

    for model_path, csv_name in [
        ("best_sbrt_model_physics.pth",    "sbrt_validation_physics.csv"),
        ("best_sbrt_model_clinical.pth",   "sbrt_validation_clinical.csv"),
        ("best_sbrt_model_diagnostic.pth", "sbrt_validation_diagnostic.csv"),
    ]:
        print(f"\n--- Evaluating: {model_path} -> {csv_name} ---")
        if not os.path.isfile(model_path):
            print(f"  WARNING: '{model_path}' not found — skipping.")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        records = []

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                inputs  = batch["image"].to(device)
                rx_gy   = float(batch["rx_gy"][0])
                pid     = batch.get("patient_id", [f"patient_{idx:03d}"])[0]

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                    outputs = sliding_window_inference(
                        inputs=inputs, roi_size=PATCH_SIZE,
                        sw_batch_size=1, predictor=model, overlap=0.25,
                    )
                outputs_gy = F.softplus(outputs.float()) * NORM_GY
                outputs_gy = outputs_gy.clamp(0.0, PHYSICAL_MAX_GY).cpu()

                ptv_dose  = outputs_gy[(inputs[:, 1:2, ...] >= 0.5).cpu().bool()]
                bl_dose   = outputs_gy[(inputs[:, 2:3, ...] <= 0.0).cpu().bool()]
                re_dose   = outputs_gy[(inputs[:, 3:4, ...] <= 0.0).cpu().bool()]
                sb_dose   = outputs_gy[(inputs[:, 4:5, ...] > 0.5).cpu().bool()]
                fr_dose   = outputs_gy[(inputs[:, 5:6, ...] > 0.5).cpu().bool()]
                fl_dose   = outputs_gy[(inputs[:, 6:7, ...] > 0.5).cpu().bool()]

                row = {"Patient_ID": pid, "Rx_Gy": rx_gy}

                # PTV metrics
                row["PTV_D95 (Gy)"] = quantile_dose(ptv_dose, 95)
                row["PTV_D98 (Gy)"] = quantile_dose(ptv_dose, 98)
                row["PTV_Mean (Gy)"] = ptv_dose.mean().item() if ptv_dose.numel() else float("nan")
                row["PTV_Max (Gy)"]  = ptv_dose.max().item()  if ptv_dose.numel() else float("nan")

                # Bladder (N0 and N+ thresholds same for 35, 31.5; differ for 28, 17.5, 14)
                for thr in (35.0, 31.5, 28.0, 17.5, 14.0):
                    row[f"Bladder_V{thr}Gy (%)"] = v_pct(bl_dose, thr)
                row["Bladder_Mean (Gy)"] = bl_dose.mean().item() if bl_dose.numel() else float("nan")

                # Anorectum
                for thr in (35.0, 31.5, 28.0, 17.5, 14.0):
                    row[f"Anorectum_V{thr}Gy (%)"] = v_pct(re_dose, thr)
                row["Anorectum_Mean (Gy)"] = re_dose.mean().item() if re_dose.numel() else float("nan")

                # Small Bowel — absolute volume (cc)
                row["SmallBowel_V28Gy (cc)"]  = v_cc(sb_dose, 28.0)  # limit: 80 cc
                row["SmallBowel_V27.5Gy (cc)"] = v_cc(sb_dose, 27.5) # limit: 2 cc

                # Femoral Heads — V14Gy (%) limit: 5%
                row["FemurR_V14Gy (%)"] = v_pct(fr_dose, 14.0)
                row["FemurL_V14Gy (%)"] = v_pct(fl_dose, 14.0)

                records.append(row)
                print(
                    f"  [{idx+1}] {pid}  Rx={rx_gy}Gy  "
                    f"D95={row['PTV_D95 (Gy)']:.2f}Gy  "
                    f"Bladder V35={row['Bladder_V35.0Gy (%)']:.1f}%  "
                    f"Anorect V35={row['Anorectum_V35.0Gy (%)']:.1f}%"
                )
                del outputs_gy, ptv_dose, bl_dose, re_dose, sb_dose, fr_dose, fl_dose
                torch.cuda.empty_cache()

        df = pd.DataFrame(records)
        float_cols = [c for c in df.columns if c not in ("Patient_ID",)]
        df[float_cols] = df[float_cols].round(2)
        df.to_csv(csv_name, index=False)
        print(f"\n  Saved '{csv_name}'")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()