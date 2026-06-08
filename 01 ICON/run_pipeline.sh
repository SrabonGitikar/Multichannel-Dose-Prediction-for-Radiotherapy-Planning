#!/bin/bash

# Pipeline Script: Data Preparation → Training → Git Push
# ======================================================

set -e  # Exit on any error

# Change to script directory so relative paths work
cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Pipeline"
echo "Working directory: $(pwd)"
echo "=========================================="

# Step 1: DICOM to nnU-Net conversion
# echo ""
# echo "[1/3] Running DICOM to nnU-Net conversion..."
python utils/preprocess.py

# Step 2: Training
# echo ""
# echo "[2/3] Starting training..."
python utils/training.py

# Step 3: Git commit and push
echo ""
echo "[3/3] Pushing to git..."
git add .
git commit -m "updated"
git push

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
