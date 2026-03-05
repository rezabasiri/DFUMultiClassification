# Environment Setup Guide

## Requirements
- Linux (tested on Ubuntu)
- Python 3.11
- NVIDIA GPU with CUDA support (RTX A5000 24GB used in development)
- CUDA 12.8+ compatible driver
- ~50GB disk space (models, data, checkpoints)

## Goal is to create environment 
"source /opt/miniforge3/bin/activate multimodal"

## Option A: Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment_multimodal.yml
conda activate multimodal

# Install PyTorch with CUDA 12.8 support
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

## Option B: venv + pip

```bash
python3.11 -m venv /venv/multimodal
source /venv/multimodal/bin/activate

# Install PyTorch with CUDA first
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements_multimodal.txt
```

## Verify Installation

```bash
python -c "
import tensorflow as tf
import torch
print(f'TF: {tf.__version__}, GPUs: {tf.config.list_physical_devices(\"GPU\")}')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import keras, sklearn, pandas, numpy, scipy, matplotlib, seaborn, PIL, cv2, tqdm
print('All core packages OK')
"
```

Expected output: TF and PyTorch both detect GPU(s), all imports succeed.

## Data Directory Structure

The project expects this layout under the repo root:

```
data/
  raw/
    DataMaster_Processed_V12_WithMissing.csv   # Master metadata CSV (~600 patients)
    bounding_box_depth.csv                      # Depth image bounding boxes
    bounding_box_thermal.csv                    # Thermal image bounding boxes
    Depth_RGB/                                  # Depth RGB images
    Depth_Map_IMG/                              # Depth map images (single-channel)
    Thermal_RGB/                                # Thermal RGB images
    Thermal_Map_IMG/                            # Thermal map images (single-channel)
```

Copy or symlink these from the original data source. The pipeline auto-creates `data/processed/` and `data/results/` at runtime.

## Quick Smoke Test

```bash
# Run with 10% data, 1 fold, CPU-only to verify pipeline works
python src/main.py --data_percentage 10 --cv_folds 1 --device-mode cpu --verbosity 2 --resume_mode fresh
```

## Production Run

```bash
python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh --cv_folds 3
```

## Key Notes
- Do NOT pass `--fold` manually. The orchestrator spawns each fold as a subprocess automatically when `cv_folds > 1`.
- All hyperparameters are in `src/utils/production_config.py`. Do not scatter config values elsewhere.
- The venv path `/venv/multimodal/` was used on the original machine. Adjust to your setup.
- If using multiple GPUs, `--device-mode multi` auto-detects and uses all available.
- For a single specific GPU: `CUDA_VISIBLE_DEVICES=0 python src/main.py --device-mode single ...`
