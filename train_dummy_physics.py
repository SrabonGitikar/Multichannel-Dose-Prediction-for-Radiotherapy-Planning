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

CHANNELS = ["0000", "0001", "0002", "0003"]  # CT, PTV, Bladder SDM, Anorectum SDM
TARGET_SPACING = (1.27, 1.27, 2.5)
PATCH_SIZE = (96, 96, 96)

PRESCRIPTION_DOSE_GY = 60.0  # Normalisation factor (Gy -> [0,1])
CONSTRAINT_CSV = os.environ.get(
    "CONSTRAINT_CSV", "./prostate_prime_constraints.csv"
)

# ===================================================================
# 2. Constraint Parsing  (Step 1)
# ===================================================================

def load_clinical_constraints(csv_path, patient_class="N0"):
    """
    Parse prostate_prime_constraints.csv into structured dicts.

    Returns
    -------
    dict  with keys:
        "v_type"  -> {"Bladder": [...], "Anorectum": [...]}
        "d_type"  -> {"PTV_max_dose_gy": float or None,
                      "PTV_coverage": [...]}
    All dose thresholds for V-Type are normalised to [0, 1] by dividing
    by PRESCRIPTION_DOSE_GY (60 Gy).
    """
    v_constraints = {"Bladder": [], "Anorectum": []}
    ptv_coverage = []
    ptv_max_dose_gy = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Patient_Class"].strip() != patient_class:
                continue

            struct = row["Structure_Name"].strip()
            ctype = row["Constraint_Type"].strip()
            metric = row["Metric"].strip()
            opt_raw = row["Optimal_Value"].strip()
            mand_raw = row["Mandatory_Value"].strip()

            # --- V-Type constraints (Bladder / Anorectum) -----------
            if ctype == "V" and struct in ("Bladder", "Anorectum"):
                # Unit must be 'pct' (fractional volume).  Skip cc-based
                if row["Unit"].strip() != "pct":
                    continue
                if not mand_raw:
                    continue  # no mandatory value -> skip

                # Parse dose threshold from metric string, e.g. "V60.4Gy"
                dose_thresh_gy = float(
                    metric.replace("V", "").replace("Gy", "")
                )
                norm_dose = dose_thresh_gy / PRESCRIPTION_DOSE_GY

                opt_v = float(opt_raw) if opt_raw else float("nan")
                mand_v = float(mand_raw)

                v_constraints[struct].append(
                    {
                        "dose_gy": dose_thresh_gy,
                        "norm_dose": norm_dose,
                        "optimal_v": opt_v,
                        "mandatory_v": mand_v,
                    }
                )

            # --- D-Type: PTV max dose --------------------------------
            if ctype == "D" and struct.startswith("PTV") and metric == "Max":
                if mand_raw:
                    ptv_max_dose_gy = float(mand_raw)

            # --- D-Type: PTV coverage (D95/D98 etc.) ------------------
            if ctype == "D" and struct.startswith("PTV") and metric.startswith("D9"):
                if mand_raw:
                    ptv_coverage.append(
                        {
                            "metric": metric,
                            "fraction": float(mand_raw),  # e.g. 0.95
                        }
                    )

    return {
        "v_type": v_constraints,
        "d_type": {
            "PTV_max_dose_gy": ptv_max_dose_gy,
            "PTV_coverage": ptv_coverage,
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
        lambda_mse=1.0,
        lambda_optimal=10.0,
        lambda_mandatory=100.0,
        lambda_ptv=5.0,
        lambda_smooth=0.1,
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
    def forward(self, pred_dose, true_dose, bladder_mask, rectum_mask, ptv_mask):
        """
        Parameters
        ----------
        pred_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        true_dose   : (B, 1, D, H, W)  — normalised [0, 1]
        bladder_mask: (B, 1, D, H, W)  — binary float
        rectum_mask : (B, 1, D, H, W)  — binary float
        ptv_mask    : (B, 1, D, H, W)  — binary float
        """
        # ------ 1. L_MSE  -------------------------------------------
        loss_mse = self.mse(pred_dose, true_dose)

        # ------ 2. L_V-Type (Dual-Tier DVH) -------------------------
        loss_optimal = torch.tensor(0.0, device=pred_dose.device,
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
                    loss_optimal = loss_optimal + viol_opt ** 2

                # Mandatory tier
                viol_mand = torch.relu(v_frac - rule["mandatory_v"])
                loss_mandatory = loss_mandatory + viol_mand ** 2

        # ------ 3. L_PTV  (D-Type coverage) -------------------------
        #  Penalise underdose: max(0, C_k - D_PTV_pred)
        #  C_k for D95 with fraction 0.95 means the dose inside PTV
        #  should be >= 0.95 (in normalised space, i.e. 95% of Rx).
        loss_ptv = torch.tensor(0.0, device=pred_dose.device,
                                dtype=pred_dose.dtype)

        ptv_n = ptv_mask.sum()
        if ptv_n > 0:
            ptv_dose = pred_dose * ptv_mask
            for rule in self.constraints["d_type"]["PTV_coverage"]:
                frac = rule["fraction"]  # e.g. 0.95 normalised
                # Underdose penalty per voxel (only inside PTV)
                underdose = torch.relu(frac - pred_dose) * ptv_mask
                loss_ptv = loss_ptv + (underdose ** 2).sum() / ptv_n

        # ------ 4. L_D-Type max dose (PTV max) ----------------------
        max_gy = self.constraints["d_type"]["PTV_max_dose_gy"]
        if max_gy is not None:
            max_norm = max_gy / PRESCRIPTION_DOSE_GY
            if ptv_n > 0:
                overdose = torch.relu(pred_dose - max_norm) * ptv_mask
                loss_ptv = loss_ptv + (overdose ** 2).sum() / ptv_n

        # ------ 5. L_smooth (Total Variation) -----------------------
        gd = pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :]
        gh = pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :]
        gw = pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1]
        loss_smooth = torch.mean(gd ** 2) + torch.mean(gh ** 2) + torch.mean(gw ** 2)

        # ------ Total -----------------------------------------------
        total = (
            self.lambda_mse * loss_mse
            + self.lambda_optimal * loss_optimal
            + self.lambda_mandatory * loss_mandatory
            + self.lambda_ptv * loss_ptv
            + self.lambda_smooth * loss_smooth
        )
        return total, {
            "mse": loss_mse.item(),
            "v_opt": loss_optimal.item(),
            "v_mand": loss_mandatory.item(),
            "ptv": loss_ptv.item(),
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


class CreateBoundaryMaskd(MapTransform):
    """
    Builds a 4-class crop_mask for RandCropByLabelClassesd.
    0=Background, 1=PTV, 2=Bladder, 3=Rectum
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        ptv = d["ch_1"] >= 0.5
        bladder = d["ch_2"] <= 0.0
        rectum = d["ch_3"] <= 0.0
        crop_mask = torch.zeros_like(d["ch_1"])
        crop_mask[rectum] = 3.0
        crop_mask[bladder] = 2.0
        crop_mask[ptv] = 1.0
        d["crop_mask"] = crop_mask
        return d


ALL_KEYS = ["ch_0", "ch_1", "ch_2", "ch_3", "dose_label"]

train_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear"),
        ),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        CreateBoundaryMaskd(keys=["ch_1"]),
        RandCropByLabelClassesd(
            keys=ALL_KEYS,
            label_key="crop_mask",
            spatial_size=PATCH_SIZE,
            num_classes=4,
            ratios=[0.0, 1.0, 1.0, 1.0],
            num_samples=4,
        ),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
        ToTensord(keys=["image", "dose_label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest", "bilinear", "bilinear", "bilinear"),
        ),
        NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
        ConcatItemsd(keys=["ch_0", "ch_1", "ch_2", "ch_3"], name="image"),
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

    # ---- Dataset ---------------------------------------------------
    print("\nFinding data...")
    data_dicts = get_data_dicts()
    print(f"Found {len(data_dicts)} patients.")

    train_files = data_dicts[:16]
    val_files = data_dicts[16:]

    cache_dir = os.path.join(DATA_DIR, "persistent_cache_physics")
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=list_data_collate,
    )

    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # ---- Model -----------------------------------------------------
    print("Building 3D U-Net...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # ---- Loss / Optimizer / Scheduler ------------------------------
    loss_function = PhysicsGuidedDoseLoss(
        constraints_dict=constraints,
        lambda_mse=10.0,        # Increased from 1.0 to anchor the model to the human ground truth
        lambda_ptv=10.0,        # Balanced equally with MSE so it actively wants to heat the tumor
        lambda_optimal=2.0,     # A mild nudge (down from 10.0)
        lambda_mandatory=25.0,  # A firm, heavy wall, but not a 100x explosion
        lambda_smooth=1.0,
        k_steepness=50.0,
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

    # ---- Training --------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}  "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

        # ---- Train -------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            inputs = batch["image"].to(device)
            targets = batch["dose_label"].to(device)

            # Normalise dose targets to [0, 1]
            normalized_targets = targets / PRESCRIPTION_DOSE_GY

            # SDM -> binary masks  (Step 3 cross-check)
            ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
                dtype=torch.float16,
            ):
                outputs = model(inputs)

                # Physics loss computed in float32 for numerical stability
                # of the sigmoid DVH approximation
                loss, components = loss_function(
                    outputs.float(),
                    normalized_targets.float(),
                    bladder_mask.float(),
                    rectum_mask.float(),
                    ptv_mask.float(),
                )

            scaler.scale(loss).backward()
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

                normalized_targets = targets / PRESCRIPTION_DOSE_GY
                ptv_mask, bladder_mask, rectum_mask = extract_binary_masks(inputs)

                loss, _ = loss_function(
                    outputs.float(),
                    normalized_targets.float(),
                    bladder_mask.float(),
                    rectum_mask.float(),
                    ptv_mask.float(),
                )
                val_loss_sum += loss.item()

                # ---- Clinical metrics (in Gy) ----------------------
                outputs_gy = outputs.float() * PRESCRIPTION_DOSE_GY

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

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_dose_model_physics.pth")
            print("  --> Saved new best model!")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
