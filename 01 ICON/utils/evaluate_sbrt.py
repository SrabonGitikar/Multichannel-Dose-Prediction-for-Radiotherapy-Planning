"""
evaluate_sbrt.py
================
Standalone re-evaluation script for a saved SBRT model checkpoint.

Corrects the PTV mask extraction bug where torch.isclose() was comparing
against the prescription dose (36.25) instead of the discrete integer value
painted by CreateDiscretePTVMapd (37.5 = rx_gy of PTV36_25 target).

Usage:
    python evaluate_sbrt.py [--config config_sbrt.yml] [--weights pe_best_dose_model_clinical_june25.pth]
"""

import os
import sys
import glob
import math
import argparse
import csv
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import yaml

from monai.data import PersistentDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
    ConcatItemsd,
    MapTransform,
    DeleteItemsd,
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SBRT Model Evaluation")
parser.add_argument(
    "--config",
    default="config_sbrt.yml",
    help="Path to the SBRT YAML config file (default: config_sbrt.yml)",
)
parser.add_argument(
    "--weights",
    default="pe_best_dose_model_clinical_june25.pth",
    help="Path to the model weights .pth file",
)
parser.add_argument(
    "--output-csv",
    default="sbrt_evaluation_results.csv",
    help="Path for the output CSV file",
)
parser.add_argument(
    "--val-split",
    type=float,
    default=0.20,
    help="Validation fraction of the dataset (default: 0.20)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(args.config, "r") as _f:
    config = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Derive constants from config
# ---------------------------------------------------------------------------
DATA_DIR    = config["dataset"]["data_dir"]
IMAGES_DIR  = os.path.join(DATA_DIR, "imagesTr")
LABELS_DIR  = os.path.join(DATA_DIR, "labelsTr")

TARGET_SPACING     = tuple(config["dataset"]["target_spacing"])
PATCH_SIZE         = tuple(config["dataset"]["patch_size"])
PRESCRIPTION_DOSE_GY = float(config["clinical_targets"]["prescription_dose_gy"])
PHYSICAL_MAX_GY    = float(config["clinical_targets"]["physical_max_gy"])

CHANNELS = [f"{ch['index']:04d}" for ch in sorted(config["channels"], key=lambda c: c["index"])]
SIB_KEYS = [lvl["name"] for lvl in config["clinical_targets"]["targets"]]
# Sorted by rx (lowest → highest) — Painter's Algorithm order for CreateDiscretePTVMapd
_TARGETS_SORTED = sorted(config["clinical_targets"]["targets"], key=lambda t: t["rx_gy"])

# Channel index helpers
_CH_INDEX   = {ch["key"]: ch["index"] for ch in config["channels"]}
_PTV_CH_IDX = next(ch["index"] for ch in config["channels"] if ch.get("role") == "PTV_binary")
_BEV_CH_IDX = next((ch["index"] for ch in config["channels"] if ch.get("role") == "BEV_Beam"), None)
_BODY_CH_IDX= next((ch["index"] for ch in config["channels"] if ch.get("role") == "Body_Mask"), None)

_SDM_ORGAN_IDX = {
    ch["organ"]: ch["index"]
    for ch in config["channels"] if ch.get("role") == "SDM"
}
_BIN_ORGAN_IDX = {
    ch["organ"]: ch["index"]
    for ch in config["channels"] if ch.get("role") == "Binary_Mask"
}

_OAR_BY_CANONICAL  = {oar["canonical"]: oar for oar in config["organs_at_risk"]}
_EXTRA_MASK_OARS   = [oar for oar in config["organs_at_risk"] if "extra_file_key" in oar]
_SDM_LOSS_OARS     = [oar for oar in config["organs_at_risk"]
                      if oar.get("requires_sdm") and "sdm_channel_key" in oar]
_BODY_HU_THRESHOLD = float(config["body_hu_threshold"])

# OAR evaluation thresholds from config
OAR_V_METRICS_GY = config["evaluation"].get("oar_v_metrics_gy", [])

# ---------------------------------------------------------------------------
# CreateDiscretePTVMapd  (identical to training.py)
# ---------------------------------------------------------------------------
class CreateDiscretePTVMapd(MapTransform):
    """
    Painter's Algorithm: paints rx_gy values (floats) into a discrete map.
    Lowest-dose volumes are painted first, highest-dose last so they always
    take precedence in overlapping regions.

    NOTE: The discrete values stored are the rx_gy floats (e.g. 36.25),
    NOT integer IDs. This is what torch.isclose() must match downstream.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.processing_order = sorted(
            [(level["name"], level["rx_gy"]) for level in config["clinical_targets"]["targets"]],
            key=lambda x: x[1],
        )

    def __call__(self, data):
        d = dict(data)
        ref_tensor = d["ch_0"]
        discrete_ptv = torch.zeros_like(ref_tensor)

        for p_key, rx_val in self.processing_order:
            if p_key in d:
                mask = d[p_key] >= 0.5
                discrete_ptv = torch.where(
                    mask,
                    torch.tensor(rx_val, device=discrete_ptv.device, dtype=discrete_ptv.dtype),
                    discrete_ptv,
                )
                del d[p_key]

        d["discrete_ptv"] = discrete_ptv
        return d


# ---------------------------------------------------------------------------
# CreateFalloffRingd  (needed for val_transforms key compatibility)
# ---------------------------------------------------------------------------
def compute_ring_mask(ptv_binary: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_data_dicts():
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.nii.gz")))
    data_dicts = []
    for label_path in label_files:
        patient_id = os.path.basename(label_path).replace(".nii.gz", "")
        pt_dict = {"dose_label": label_path}
        for i, ch in enumerate(CHANNELS):
            pt_dict[f"ch_{i}"] = os.path.join(IMAGES_DIR, f"{patient_id}_{ch}.nii.gz")

        # Extra mask files (bowel, femur, …)
        for oar in _EXTRA_MASK_OARS:
            key    = oar["extra_file_key"]
            suffix = oar.get("file_suffix", key)
            pt_dict[key] = os.path.join(IMAGES_DIR, f"{patient_id}_{suffix}.nii.gz")

        # Individual PTV NIfTI files for CreateDiscretePTVMapd
        ptv_files = glob.glob(os.path.join(IMAGES_DIR, f"{patient_id}_PTV*.nii.gz"))
        for p_file in ptv_files:
            _bn      = os.path.basename(p_file).replace(".nii.gz", "")
            key_name = _bn[len(patient_id) + 1:]
            pt_dict[key_name] = p_file

        data_dicts.append(pt_dict)
    return data_dicts


# Build the full key list for MONAI transforms
EXTRA_MASK_KEYS = [oar["extra_file_key"] for oar in _EXTRA_MASK_OARS]
ALL_KEYS = (
    [f"ch_{i}" for i in range(len(CHANNELS))]
    + EXTRA_MASK_KEYS
    + ["dose_label"]
    + SIB_KEYS
)

# Spacing modes: one per ALL_KEYS entry
_SPACING_MODES = (
    tuple(ch["interpolation"] for ch in sorted(config["channels"], key=lambda c: c["index"]))
    + tuple("nearest" for _ in EXTRA_MASK_KEYS)
    + ("bilinear",)                         # dose_label
    + tuple("nearest" for _ in SIB_KEYS)   # individual PTV masks
)

# ConcatItemsd key order must match the model's expected channel layout
CONCAT_KEYS = ["ch_0", "discrete_ptv"] + [
    f"ch_{i}" for i in range(2, len(CHANNELS))
]

val_transforms = Compose([
    LoadImaged(keys=ALL_KEYS, allow_missing_keys=True),
    EnsureChannelFirstd(keys=ALL_KEYS, allow_missing_keys=True),
    Spacingd(
        keys=ALL_KEYS,
        pixdim=TARGET_SPACING,
        mode=_SPACING_MODES,
        allow_missing_keys=True,
    ),
    CreateDiscretePTVMapd(keys=["ch_0"]),
    NormalizeIntensityd(keys=["ch_0"], nonzero=False, channel_wise=True),
    ConcatItemsd(keys=CONCAT_KEYS, name="image"),
    ToTensord(keys=["image", "dose_label"] + EXTRA_MASK_KEYS),
])


# ---------------------------------------------------------------------------
# DVH helpers
# ---------------------------------------------------------------------------
def quantile_dose(dose_1d: torch.Tensor, pct: float) -> float:
    """D{pct}: dose exceeded by pct% of the volume.  D95 → pct=95 → q=0.05."""
    if dose_1d.numel() == 0:
        return float("nan")
    q = 1.0 - pct / 100.0
    return torch.quantile(dose_1d.float(), q).item()


def v_metric(dose_1d: torch.Tensor, threshold_gy: float) -> float:
    """Volume fraction (%) receiving > threshold_gy Gy."""
    if dose_1d.numel() == 0:
        return float("nan")
    return ((dose_1d > threshold_gy).float().mean() * 100.0).item()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    # --- Logging -----------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/evaluate_sbrt_{ts}.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Config        : {args.config}")
    log.info(f"Weights file  : {args.weights}")
    log.info(f"Output CSV    : {args.output_csv}")
    log.info(f"Rx dose       : {PRESCRIPTION_DOSE_GY} Gy")
    log.info(f"Physical max  : {PHYSICAL_MAX_GY} Gy")

    # --- Dataset split -----------------------------------------------------
    data_dicts = get_data_dicts()
    n_total    = len(data_dicts)
    log.info(f"Found {n_total} patients in {IMAGES_DIR}")

    n_val      = max(1, round(n_total * args.val_split))
    n_train    = n_total - n_val
    val_files  = data_dicts[n_train:]
    log.info(f"Using the last {n_val} patients as validation set")

    # --- Persistent dataset + loader --------------------------------------
    cache_dir = os.path.join(DATA_DIR, "persistent_cache_physics")
    os.makedirs(cache_dir, exist_ok=True)

    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=False,
    )

    # --- Build model -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = UNet(
        spatial_dims=3,
        in_channels=len(CHANNELS),
        out_channels=1,
        channels=tuple(config["model"]["channels"]),
        strides=tuple(config["model"]["strides"]),
        num_res_units=int(config["model"].get("num_res_units", 2)),
    ).to(device)

    # --- Load weights ------------------------------------------------------
    weights_path = args.weights
    if not os.path.isfile(weights_path):
        log.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    log.info(f"Loaded weights from '{weights_path}'")
    model.eval()

    # SIB target names from config (for labelling CSV columns)
    sib_target_names = [lvl["name"] for lvl in config["clinical_targets"]["targets"]]
    primary_name = max(config["clinical_targets"]["targets"], key=lambda t: t["rx_gy"])["name"]

    # Bladder / Rectum SDM channel indices
    _bladder_key = next(
        (oar["canonical"] for oar in config["organs_at_risk"] if oar.get("sdm_channel_key") == "ch_2"),
        "Bladder",
    )
    _rectum_key = next(
        (oar["canonical"] for oar in config["organs_at_risk"] if oar.get("sdm_channel_key") == "ch_3"),
        "Anorectum",
    )

    # --- Evaluation loop ---------------------------------------------------
    records = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            inputs  = batch["image"].to(device)
            targets = batch["dose_label"].to(device)

            # Sliding-window inference with AMP
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                )

            # Guard against NaN/Inf
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
            outputs_gy = F.softplus(outputs.float()) * PRESCRIPTION_DOSE_GY
            outputs_gy = torch.clamp(outputs_gy, min=0.0, max=PHYSICAL_MAX_GY)

            # Apply body mask to suppress ghost radiation
            if _BODY_CH_IDX is not None:
                body_mask_eval = (inputs[:, _BODY_CH_IDX:_BODY_CH_IDX+1, ...] > 0.5).float()
                outputs_gy = outputs_gy * body_mask_eval

            # Build organ masks from SDM channels
            bladder_mask = (inputs[:, _SDM_ORGAN_IDX[_bladder_key]:_SDM_ORGAN_IDX[_bladder_key]+1, ...] <= 0.0).float()
            rectum_mask  = (inputs[:, _SDM_ORGAN_IDX[_rectum_key]:_SDM_ORGAN_IDX[_rectum_key]+1, ...] <= 0.0).float()

            # Discrete PTV channel (CPU) — contains the painted rx_gy floats
            discrete_ptv = inputs[:, _PTV_CH_IDX:_PTV_CH_IDX+1, ...].cpu()

            # --- CPU offloading (memory management) -----------------------
            outputs_gy_cpu      = outputs_gy.cpu()
            bladder_mask_cpu    = bladder_mask.cpu()
            rectum_mask_cpu     = rectum_mask.cpu()

            # Extra mask tensors (bowel, femur, …)
            extra_masks_cpu = {}
            for oar in _EXTRA_MASK_OARS:
                key = oar["extra_file_key"]
                if key in batch:
                    extra_masks_cpu[key] = batch[key].cpu()

            # --- Patient ID -----------------------------------------------
            try:
                label_path = val_files[idx]["dose_label"]
                patient_id = os.path.basename(label_path).replace(".nii.gz", "")
            except (IndexError, KeyError):
                patient_id = f"patient_{idx:03d}"

            row = {"Patient_ID": patient_id}

            # --- SIB PTV metrics (ROBUST: bypass float-matching entirely) ----
            # Print unique discrete values on first patient to verify cache state
            unique_vals = torch.unique(discrete_ptv)
            if idx == 0:
                print(f"  [Debug] Unique values in PTV channel: {unique_vals.tolist()}")
                log.info(f"  [Debug] Unique values in PTV channel: {unique_vals.tolist()}")

            # Grab ALL PTV voxels by ignoring background (0.0).
            # This is robust to float precision and stale cache issues since
            # we no longer depend on a specific painted value.
            ptv_mask_all = (discrete_ptv > 0.5)
            ptv_dose_all = outputs_gy_cpu[ptv_mask_all]

            # Report under the primary target name (unified single-PTV SBRT)
            if ptv_dose_all.numel() > 0:
                row[f"{primary_name}_D95 (Gy)"]  = quantile_dose(ptv_dose_all, 95)
                row[f"{primary_name}_D98 (Gy)"]  = quantile_dose(ptv_dose_all, 98)
                row[f"{primary_name}_Mean (Gy)"] = ptv_dose_all.mean().item()
                row[f"{primary_name}_Max (Gy)"]  = ptv_dose_all.max().item()

                # ICRU 83 Homogeneity Index
                d2  = quantile_dose(ptv_dose_all, 2)
                d98 = quantile_dose(ptv_dose_all, 98)
                d50 = torch.median(ptv_dose_all.float()).item()
                row[f"{primary_name}_HI"] = round((d2 - d98) / d50, 4) if d50 > 0 else float("nan")
            else:
                log.warning(f"  WARNING: {patient_id} — PTV channel is all zeros! Check cache.")
                for col in ["_D95 (Gy)", "_D98 (Gy)", "_Mean (Gy)", "_Max (Gy)", "_HI"]:
                    row[f"{primary_name}{col}"] = float("nan")

            # --- Bladder --------------------------------------------------
            bladder_dose = outputs_gy_cpu[bladder_mask_cpu.bool()]
            row["Bladder_Mean (Gy)"] = bladder_dose.mean().item() if bladder_dose.numel() else float("nan")
            row["Bladder_Max (Gy)"]  = bladder_dose.max().item()  if bladder_dose.numel() else float("nan")
            for thresh in OAR_V_METRICS_GY:
                row[f"Bladder_V{thresh}Gy (%)"] = v_metric(bladder_dose, thresh)

            # --- Rectum ---------------------------------------------------
            rectum_dose = outputs_gy_cpu[rectum_mask_cpu.bool()]
            row["Rectum_Mean (Gy)"] = rectum_dose.mean().item() if rectum_dose.numel() else float("nan")
            row["Rectum_Max (Gy)"]  = rectum_dose.max().item()  if rectum_dose.numel() else float("nan")
            for thresh in OAR_V_METRICS_GY:
                row[f"Rectum_V{thresh}Gy (%)"] = v_metric(rectum_dose, thresh)

            # --- Extra masks (bowel, femur, …) ----------------------------
            for oar in _EXTRA_MASK_OARS:
                key      = oar["extra_file_key"]
                canonical= oar["canonical"]
                if key in extra_masks_cpu:
                    oar_dose = outputs_gy_cpu[extra_masks_cpu[key].cpu().bool()]
                    row[f"{canonical}_Mean (Gy)"] = oar_dose.mean().item() if oar_dose.numel() else float("nan")
                    row[f"{canonical}_Max (Gy)"]  = oar_dose.max().item()  if oar_dose.numel() else float("nan")

            # --- Global Dmax ----------------------------------------------
            row["Dmax (Gy)"] = outputs_gy_cpu.max().item()

            # --- Print summary row ----------------------------------------
            log.info(
                f"  [{idx+1}/{len(val_loader)}] {patient_id}  "
                f"{primary_name}_D95={row.get(f'{primary_name}_D95 (Gy)', float('nan')):.2f} Gy  "
                f"Bladder Mean={row['Bladder_Mean (Gy)']:.2f} Gy  "
                f"Rectum Mean={row['Rectum_Mean (Gy)']:.2f} Gy  "
                f"Dmax={row['Dmax (Gy)']:.2f} Gy"
            )

            records.append(row)

            # --- Aggressively free memory ---------------------------------
            del outputs_gy_cpu, discrete_ptv, bladder_dose, rectum_dose
            del bladder_mask_cpu, rectum_mask_cpu, extra_masks_cpu
            del outputs, outputs_gy, bladder_mask, rectum_mask
            del inputs, targets
            torch.cuda.empty_cache()

    # --- Save CSV ----------------------------------------------------------
    df = pd.DataFrame(records)
    float_cols = [c for c in df.columns if c != "Patient_ID"]
    df[float_cols] = df[float_cols].round(2)
    df.to_csv(args.output_csv, index=False)
    log.info(f"\nSaved results to '{args.output_csv}'")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
