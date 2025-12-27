# RTX 5090 Compatibility Fix

## Problem
RTX 5090 GPUs have compute capability 12.0, which requires TensorFlow 2.17+ with CUDA 12.x support.

Current error:
```
CUDA_ERROR_INVALID_PTX
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0
```

## Solution

### Step 1: Verify CUDA Version
```bash
nvidia-smi
nvcc --version
```

Ensure you have CUDA 12.x or 13.x installed.

### Step 2: Reinstall TensorFlow 2.18.1 (CUDA 13 compatible)

```bash
# Uninstall old TensorFlow
pip uninstall -y tensorflow tensorflow-gpu tf-keras keras

# Install TensorFlow 2.18.1 with CUDA 13 support
pip install tensorflow==2.18.1

# Or reinstall all requirements
pip install -r requirements_cu13.txt --upgrade
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'CUDA available: {tf.test.is_built_with_cuda()}'); print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

Expected output:
```
TensorFlow: 2.18.1
CUDA available: True
GPUs: 2
```

### Step 4: Test GPU Configuration

```bash
python agent_communication/gpu_setup/test_gpu_config.py
```

All tests should pass without CUDA_ERROR_INVALID_PTX errors.

### Step 5: Run Training

```bash
# Single GPU test
python src/main.py --device-mode single

# Multi-GPU test
python src/main.py --device-mode multi
```

## Notes

- TensorFlow 2.18.1+ includes pre-compiled kernels for compute capability 12.0 (RTX 5090)
- If you still see PTX warnings, they should not cause failures (JIT compilation will succeed)
- The warnings "could take 30 minutes" only apply to first-run kernel compilation (cached after)

## Alternative: Use TensorFlow Nightly (if 2.18.1 doesn't work)

```bash
pip uninstall -y tensorflow
pip install tf-nightly
```
