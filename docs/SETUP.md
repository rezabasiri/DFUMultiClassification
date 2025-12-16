# Environment Setup Guide

This guide provides instructions for setting up the environment for the DFU Multi-Classification project.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- CUDA 11.8 or 12.1 (for GPU acceleration)
- 16GB+ RAM (32GB recommended)
- 10GB+ disk space

## Installation Options

### Option 1: Using pip (Virtual Environment)

1. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using Conda (Recommended)

1. **Create conda environment from file:**
```bash
conda env create -f environment.yml
```

2. **Activate the environment:**
```bash
conda activate dfu-classification
```

### Option 3: Manual Installation

If you encounter issues with the automated methods:

```bash
# Core packages
pip install tensorflow==2.13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Data processing
pip install pandas numpy pillow scikit-learn imbalanced-learn

# Visualization
pip install matplotlib seaborn

# Generative models
pip install diffusers transformers

# Model interpretability
pip install shap

# Utilities
pip install tqdm opencv-python
```

## Verify Installation

Run the following to verify your installation:

```python
import tensorflow as tf
import torch
import pandas as pd
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available (TF): {tf.config.list_physical_devices('GPU')}")
print(f"GPU available (PyTorch): {torch.cuda.is_available()}")
```

## Configuration

1. **Update paths in `src/utils/config.py`:**
   - Set your data directory path
   - Configure result directory for outputs

2. **Prepare data structure:**
```
data/
├── raw/
│   ├── Depth_Map_IMG/
│   ├── Depth_RGB/
│   ├── Thermal_Map_IMG/
│   ├── Thermal_RGB/
│   ├── DataMaster_Processed_V12_WithMissing.csv
│   ├── bounding_box_depth.csv
│   └── bounding_box_thermal.csv
└── processed/
    └── best_matching.csv (auto-generated)
```

## Running the Project

1. **Train a model:**
```bash
cd src
python main.py
```

2. **With custom parameters:**
```bash
python main.py --data-percentage 100 --n-runs 3
```

## Troubleshooting

### GPU Not Detected

**TensorFlow:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If empty, reinstall with:
```bash
pip install tensorflow[and-cuda]
```

**PyTorch:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

If encountering OOM errors:
- Reduce `global_batch_size` in `src/main.py`
- Enable mixed precision training
- Use gradient checkpointing

### Import Errors

Ensure you're running from the project root and the src directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/DFUMultiClassification"
```

## Hardware Recommendations

### Minimum Requirements
- CPU: 4+ cores
- RAM: 16GB
- GPU: 6GB VRAM (e.g., RTX 2060)
- Storage: 20GB

### Recommended
- CPU: 8+ cores
- RAM: 32GB
- GPU: 12GB+ VRAM (e.g., RTX 3080, A100)
- Storage: 50GB SSD

## Additional Resources

- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation](https://developer.nvidia.com/cuda-downloads)
