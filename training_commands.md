# Prostate Dose Prediction Training Pipeline

To run the end-to-end training pipeline, open your terminal and execute the following commands one by one.

### 1. Create and Activate the Virtual Environment
First, create a new Python virtual environment and activate it. Then, install all the required packages (like MONAI, PyTorch, pydicom, etc.) using the provided requirements file.
```bash
python -m venv dose_env
source dose_env/bin/activate
pip install -r requirements.txt
```

### 2. Audit Patient Data (Optional)
Check that all 11 patients have the correct complete set of DICOM files (CT, RTStruct, RTDose, RTPlan).
```bash
python audit_patients.py
```

### 3. Convert DICOM Data
Convert the raw DICOM files into NIfTI format (the format expected by MONAI and nnU-Net). This will output to `nnUNet_raw/Dataset001_ProstateDose`.
```bash
python dicom_to_nnunet.py
```

### 4. Check Plumbed Data (Optional)
This script is a sanity check to verify the PyTorch data loaders and transformations are working correctly before training begins.
```bash
python Data_plumbing.py
```

### 5. Train the 3D U-Net Model
Run the primary training script. This script loads the NIfTI files, applies the physics-guided loss function (incorporating PTV, Bladder, and Rectum doses), and caches the data for speed. The best model weights will be saved to `best_dose_model.pth`.
```bash
python train_monai.py
```

### 6. Run Inference
Once training is complete, you can use the trained model to generate predictions using sliding window inference.
```bash
python inference.py
python inference.py --patient prostate_000 --output-dir data/output/test3 --model best_dose_model_physics.pth
```

```bash
python utils/nifti_to_rtdose.py \
    --ct-rs-dir "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/1de3b35a.8614.43b1.8bb4.37877ce504dd" \
    --nifti-path "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/data/output/test31/prostate_000_predicted_dose.nii.gz" \
    --dose-spacing 2.5
```

### 7. Visualize Predictions (Optional)
Plot masks and predictions to visually verify the outputs.
```bash
python plot_masks.py
```


```bash
python utils/pipeline.py \
    --dicom-dir "/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/raw_dicom_nifti/f40d8cf2.f057.409e.a9cb.434ea5aa8eed/1.2.826.0.1.3680043.10.1561.260.1390.924/" \
    --model best_dose_model_physics.pth \
    --dose-spacing 2.5

```

/mnt/nvme/nvme-2TB-storage/sougata/python/Multichannel-Dose-Prediction-for-Radiotherapy-Planning/testdata/f40d8cf2.f057.409e.a9cb.434ea5aa8eed/1.2.826.0.1.3680043.10.1561.260.1390.924/