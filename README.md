````markdown id="f2m8vx"
# Physics-Guided Deep Learning for Prostate Radiotherapy Dose Prediction

DATA_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/data/Prostate PRIME Standard arm d69"
OUTPUT_DIR = "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/nnUNet_raw/Dataset001_ProstateDose"
DATASET_NAME = "Dataset001_ProstateDose"

## Overview

This project focuses on developing a deep learning framework for predicting three-dimensional radiotherapy dose distributions for prostate cancer treatment planning.

The pipeline uses a 3D U-Net implemented using the MONAI framework and is trained on processed clinical radiotherapy datasets containing:

- CT scans,
- Planning Target Volume (PTV) masks,
- Bladder Signed Distance Maps (SDMs),
- Anorectum Signed Distance Maps (SDMs),
- and clinically delivered radiation dose distributions.

The long-term objective of the project is to explore physics-guided deep learning strategies for radiotherapy treatment planning.

---

# Project Motivation

Modern radiotherapy planning is a highly complex optimization problem involving:

- accurate tumor dose coverage,
- sparing of healthy organs-at-risk (OARs),
- spatial dose falloff,
- and clinically meaningful dose distributions.

Traditional voxel-wise regression losses such as Mean Squared Error (MSE) do not explicitly encode these physical and clinical constraints.

This project therefore aims to investigate hybrid approaches combining:

- deep learning,
- geometric anatomical priors,
- and physics-guided optimization strategies.

---

# Current Pipeline

## Input Channels

For each patient, the model uses four aligned volumetric input channels:

| Channel | Description |
|---|---|
| 0000 | CT Scan |
| 0001 | Planning Target Volume (PTV) |
| 0002 | Bladder Signed Distance Map (SDM) |
| 0003 | Anorectum Signed Distance Map (SDM) |

The target output is the clinically delivered 3D radiation dose distribution.

---

# Data Preprocessing

The preprocessing pipeline performs:

- DICOM/NIfTI loading,
- channel formatting,
- voxel spacing resampling,
- CT intensity normalization,
- multi-channel tensor concatenation,
- and volumetric patch extraction.

All patient volumes are resampled to a uniform voxel spacing:

```python
TARGET_SPACING = (1.27, 1.27, 2.5)
````

Training is performed using random 3D patches:

```python
PATCH_SIZE = (96, 96, 96)
```

---

# Model Architecture

The current implementation uses a MONAI-based 3D U-Net architecture:

```python
UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

The network predicts a continuous-valued radiation dose distribution.

---

# Loss Function

The baseline implementation currently uses voxel-wise Mean Squared Error (MSE):

```python
loss = MSE(predicted_dose, true_dose)
```

Future work aims to extend this toward physics-guided losses incorporating:

* PTV coverage constraints,
* organ-at-risk penalties,
* smoothness regularization,
* distance-aware weighting,
* and DVH-based optimization.

---

# Inference Pipeline

Inference is performed using MONAI sliding-window inference to handle large 3D patient volumes efficiently.

The workflow:

1. Loads the trained model,
2. Applies preprocessing transforms,
3. Performs patch-wise volumetric inference,
4. Reconstructs the full dose distribution,
5. Saves predicted dose volumes in NIfTI format.

---

# Clinical Evaluation Metrics

Current validation metrics include:

* Validation Mean Squared Error (MSE),
* PTV D95,
* Mean Bladder Dose,
* Mean Rectum Dose.

These metrics provide clinically interpretable measures of:

* target coverage,
* and organ-at-risk sparing.

---

# Future Directions

Planned future developments include:

* Physics-guided loss formulations,
* Dose Volume Histogram (DVH)-based optimization,
* Beam geometry integration,
* Attention-based architectures,
* Improved volumetric sampling strategies,
* and clinically informed evaluation pipelines.

---

# Frameworks and Libraries

This project currently uses:

* PyTorch
* MONAI
* NumPy
* SimpleITK

---

# Research Context

This work is being developed as part of a research internship in deep learning-assisted radiation treatment planning.

The project lies at the intersection of:

* medical physics,
* deep learning,
* optimization,
* and computational radiotherapy.

---

# Disclaimer

This project is currently intended for research and educational purposes only.

It is not validated for clinical deployment or patient treatment use.

```
```


### info

python utils/inference_pipeline.py     --dicom-dir "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/test_trained_model_28_05_2026/08fdb709.60a5.4b9e.9dab.68163167ca7c/1.2.826.0.1.3680043.10.1561.939.2268.249/"     --config config/config.yml     --model model/best_dose_model_clinical_june10.pth

==================================================
  RADIOTHERAPY DOSE PREDICTION PIPELINE
==================================================

[1] Workspace created: /tmp/dose_infer_xnnp639_

[2] Preprocessing DICOM...
  [preprocess] CT loaded: shape=(240, 512, 512)  spacing=(1.269531, 1.269531, 2.5)
  [preprocess] RTSTRUCT: /mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/test_trained_model_28_05_2026/08fdb709.60a5.4b9e.9dab.68163167ca7c/1.2.826.0.1.3680043.10.1561.939.2268.249/1.2.826.0.1.3680043.10.1561.939.2268.249.1.8608657.322-RTSTRUCT.dcm
  [preprocess] PTV (6 structures): ['PTV_44/20', 'PTV_62/20', 'PTV_62/20 PLAN', 'PTV_62/20 PLAN1', 'PTV_44/20 PLAN', 'PTV_44/20 PLAN1']
  [preprocess] Bladder: Bladder
  [preprocess] Anorectum: Anorectum
  [preprocess] Body: BODY
  [preprocess] Penile_Bulb: PenileBulb
  [preprocess] Bag_Bowel: Bag_Bowel
  [preprocess] Femur: L=Femur_Head_L R=Femur_Head_R
  [pipeline] Gantry Angles: [179.0]
  [pipeline] PTV isocenter (mm): [-8.  -1.8 19.9]
  [pipeline] BEV voxels: 8,128,706
  [preprocess] Channels saved → /tmp/dose_infer_xnnp639_/imagesTr

[3] Running Neural Network Inference...

[inference] Loading model on cuda ...
[inference] Weights loaded from model/best_dose_model_clinical_june10.pth
[inference] Native CT  : size=(512, 512, 240)  spacing=(1.269531011581421, 1.269531011581421, 2.5)
[inference] Resampled  : size=[512, 512, 240]  spacing=(1.27, 1.27, 2.5)
[inference] Input tensor shape : (1, 7, 512, 512, 240)  (B, 7, D, H, W) — at target spacing
Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /pytorch/torch/csrc/autograd/python_variable_indexing.cpp:347.)
Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /pytorch/torch/csrc/autograd/python_variable_indexing.cpp:347.)
[inference] Body-masked output shape : (1, 1, 512, 512, 240)
[inference] Prediction complete.
[inference]   Shape  : (512, 512, 240)  (at target_spacing)
[inference]   Dose   : [0.00, 63.01] Gy
[inference] Saved NIfTI → /tmp/dose_infer_xnnp639_/case_infer_predicted_dose.nii.gz

[4] Building RTDOSE DICOM...
[nifti_to_rtdose] CT: 512×512×240  spacing=(1.270,1.270,2.500) mm
[nifti_to_rtdose] Patient: Anonymous PatientName | ID: 08fdb709.60a5.4b9e.9dab.68163167ca7c
[nifti_to_rtdose] RTSTRUCT: 1.2.826.0.1.3680043.10.1561.939.2268.249.1.8608657.322
[nifti_to_rtdose] Dose grid: 260×260×240  @ 2.5 mm  (~64.9 MB)
[nifti_to_rtdose] Dose resampled: (240, 260, 260)  [0.00, 52.45] Gy
invalid value encountered in cast
[nifti_to_rtdose] Linked RTPLAN: 1.2.826.0.1.3680043.8.498.75760465215771516442147367051732334774
[nifti_to_rtdose] Saved: /mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/test_trained_model_28_05_2026/08fdb709.60a5.4b9e.9dab.68163167ca7c/1.2.826.0.1.3680043.10.1561.939.2268.249/predicted-dose-20260616_141956.dcm
[nifti_to_rtdose] FrameOfReferenceUID : 1.2.826.0.1.3680043.10.1561.939.2268.249.1.76650
[nifti_to_rtdose] StudyInstanceUID    : 1.2.826.0.1.3680043.10.1561.939.2268.249
[nifti_to_rtdose] DoseGridScaling     : 1.221086e-08 Gy/count

[5] Cleaning up workspace...

==================================================
  PIPELINE COMPLETE
==================================================
  RTDOSE saved to: /mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/test_trained_model_28_05_2026/08fdb709.60a5.4b9e.9dab.68163167ca7c/1.2.826.0.1.3680043.10.1561.939.2268.249/predicted-dose-20260616_141956.dcm
(dose_env) (base) sougata@chavi-dell-precision:/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/01 ICON$ 