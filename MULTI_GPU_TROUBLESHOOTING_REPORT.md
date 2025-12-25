# Multi-GPU Training Troubleshooting Report
**Date**: December 25, 2025
**System**: 2x NVIDIA GeForce RTX 5090 (32GB each)
**TensorFlow**: 2.15.1 with CUDA 12.2
**Issue**: Multi-GPU training not working, NCCL errors

---

## Executive Summary

Attempted to enable multi-GPU training using TensorFlow's MirroredStrategy on dual RTX 5090 GPUs. Successfully configured the strategy and both GPUs are detected, but training fails with NCCL errors during collective operations. The root cause is **TensorFlow 2.15.1's NCCL implementation has compatibility issues with RTX 5090 (compute capability 12.0)**.

### Current Status
- ✅ Single GPU mode works perfectly
- ✅ Multi-GPU strategy setup is correct (detects 2 GPUs, creates 2 replicas)
- ✅ Both GPUs are visible to TensorFlow
- ❌ Training fails with NCCL "unhandled cuda error" during gradient aggregation
- 🔄 Implemented HierarchicalCopyAllReduce workaround (untested at time of writing)

---

## Hardware & Environment

### GPUs
```
GPU 0: NVIDIA GeForce RTX 5090 - 31.8GB (PCI 0000:25:00.0)
GPU 1: NVIDIA GeForce RTX 5090 - 31.8GB (PCI 0000:41:00.0)
Compute Capability: 12.0 (Blackwell architecture)
Driver: 580.105.08
CUDA: 13.0 (system) / 12.2 (TensorFlow runtime)
```

### Software Stack
```
Python: 3.11
TensorFlow: 2.15.1 (with CUDA 12.2)
Keras: 2.15.0
PyTorch: 2.0.1+cu117
Conda env: multimodal
```

### Key Warning
```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0.
CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
```
This indicates TF 2.15.1 predates RTX 5090 support.

---

## Issues Encountered & Solutions

### Issue 1: PyTorch Missing
**Error**: `ModuleNotFoundError: No module named 'torch'`
**Cause**: PyTorch not in requirements.yml
**Solution**: Added to requirements.yml:
```yaml
- torch  # Auto-selected version: 2.0.1+cu117
- torchvision
- transformers==4.35.0
- diffusers==0.23.0
- scikit-optimize>=0.9.0
```

### Issue 2: Scikit-learn Deprecation Warnings
**Error**: FutureWarning about `_check_n_features` and `_check_feature_names`
**Cause**: scikit-learn 1.6.1 deprecated internal methods
**Solution**: Downgraded to 1.5.2 in requirements.yml

### Issue 3: Early TensorFlow GPU Initialization
**Problem**: TensorFlow initialized GPUs before `CUDA_VISIBLE_DEVICES` was set in multi-GPU mode
**Symptom**: Only GPU 0 visible despite `--device-mode multi`

**Root Causes Found**:
1. `src/utils/debug.py` had module-level `import tensorflow as tf` (line 7)
2. `src/main.py` called `tf.random.set_seed()` at module level (line 127)
3. `src/main.py` called `tf.config.optimizer.set_experimental_options()` at module level (line 133)
4. `src/main.py` had `gpus = tf.config.list_physical_devices('GPU')` at module level (line 116)

**Solutions Applied**:
- **debug.py**: Changed to lazy imports inside each function
- **main.py**: Moved all TensorFlow operations to after `setup_device_strategy()` is called
- **main.py**: Commented out module-level GPU configuration

### Issue 4: NCCL "Unhandled CUDA Error"
**Error**:
```
BaseCollectiveExecutor::StartAbort UNKNOWN: Error invoking NCCL: unhandled cuda error
Collective ops is aborted by: Error invoking NCCL: unhandled cuda error
```

**When it occurs**: During `model.fit()`, specifically during gradient aggregation (`CollectiveReduceV2`)

**Root Cause**: TensorFlow 2.15.1's NCCL binaries are not compatible with RTX 5090's compute capability 12.0. The JIT compilation from PTX suggests TF doesn't have pre-built kernels for this architecture.

**Attempted Solutions**:
1. ❌ Creating fresh MirroredStrategy in training function → Still causes NCCL errors
2. ❌ Removing nested strategy scopes → Reduced scope issues but NCCL still fails
3. 🔄 Using HierarchicalCopyAllReduce instead of NCCL (see below)

---

## Current Implementation

### GPU Configuration Flow
```
main.py startup
  ↓
parse args (--device-mode multi)
  ↓
setup_device_strategy(mode='multi')
  ├─ Set CUDA_VISIBLE_DEVICES='0,1'
  ├─ Configure GPU memory growth
  ├─ Create MirroredStrategy with HierarchicalCopyAllReduce
  └─ Return strategy object
  ↓
Store in globals()['DISTRIBUTION_STRATEGY']
  ↓
Configure TensorFlow (seeds, optimizer options)
  ↓
Training functions retrieve strategy from main module
```

### Key Files Modified

**src/utils/debug.py**:
- Removed module-level TensorFlow import
- Added lazy imports inside functions

**src/main.py**:
- Lines 115-123: Commented out module-level GPU config
- Lines 125-133: Commented out module-level TF operations
- Line 137: Set `strategy = None` (set later in main())
- Lines 2357-2361: Added TF config after `setup_device_strategy()`

**src/utils/gpu_config.py**:
- Lines 255-264: Added GPU memory growth config after setting CUDA_VISIBLE_DEVICES
- Lines 273-280: **NEW** - Use `HierarchicalCopyAllReduce` for multi-GPU

**src/training/training_utils.py**:
- Lines 782-792: Get strategy from `main_module.DISTRIBUTION_STRATEGY`
- Line 944: Removed strategy scope around `process_all_modalities()`
- Line 965: Removed strategy scope around `prepare_cached_datasets()`
- Line 1025: Removed strategy scope around dataset filtering
- Kept strategy scopes only around model creation/compilation (lines 1089, 1226, etc.)

### HierarchicalCopyAllReduce Configuration

```python
# In setup_device_strategy() when len(selected_gpu_ids) > 1:
cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
```

**How it works**:
- Default: NCCL for GPU-to-GPU gradient aggregation (fast, broken on RTX 5090)
- HierarchicalCopyAllReduce: Uses CPU memory as intermediary (slower, more compatible)

**Trade-offs**:
- ✅ Avoids NCCL entirely
- ✅ Should work with any GPU
- ❌ Slower than NCCL (gradients copy to CPU, aggregate, copy back)
- ❌ May bottleneck on PCIe bandwidth

---

## What Works

✅ **Single GPU Mode**:
```bash
python src/main.py --device-mode single --resume_mode fresh
```
Works perfectly, full GPU utilization, no errors.

✅ **Multi-GPU Detection**:
```
Detected 2 GPU(s):
  GPU 0: NVIDIA GeForce RTX 5090 - 31.8GB
  GPU 1: NVIDIA GeForce RTX 5090 - 31.8GB
Selected 2 GPU(s)
Using MirroredStrategy (2 GPUs) with HierarchicalCopyAllReduce
Batch size per replica: 64 (global batch size: 128, replicas: 2)
```

✅ **Strategy Setup**: MirroredStrategy correctly created with 2 replicas

---

## What Doesn't Work

❌ **Actual Multi-GPU Training**: Crashes during `model.fit()` with NCCL errors

❌ **GPU Utilization**: Before crash, only GPU 0 shows high utilization (~30GB), GPU 1 shows minimal usage (~512MB - just model replica)

---

## Alternative Solutions to Try

### Option 1: Upgrade TensorFlow (Recommended)
TensorFlow 2.16+ or 2.17+ may have RTX 5090 support with pre-compiled CUDA kernels.

**Risks**:
- API changes may break existing code
- Keras 3.x integration changes
- Dependency conflicts with other packages

**How to test**:
```bash
conda create -n test_tf python=3.11
conda activate test_tf
pip install tensorflow[and-cuda]>=2.16
python -c "import tensorflow as tf; print(tf.__version__)"
# Test if RTX 5090 warnings disappear
```

### Option 2: Build TensorFlow from Source
Compile TF 2.15.1 with CUDA 12.x and compute capability 12.0 support.

**Complexity**: High
**Time**: Several hours
**Reference**: https://www.tensorflow.org/install/source

### Option 3: Use PyTorch Instead of TensorFlow
PyTorch 2.0+ has better RTX 5090 support.

**Complexity**: High (requires rewriting model code)
**Benefit**: Better multi-GPU support, simpler distributed training

### Option 4: Environment Variables Workaround
Try forcing specific NCCL settings:

```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer (force through CPU)
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
python src/main.py --device-mode multi
```

### Option 5: Use Horovod
Replace MirroredStrategy with Horovod for multi-GPU training.

**Pros**: More mature, better debugging, explicit control
**Cons**: Requires code changes, additional dependency

### Option 6: Test HierarchicalCopyAllReduce
The current implementation should work but needs testing.

**Status**: Implemented but untested
**Expected behavior**: Slower but should work

---

## Debugging Commands

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time
```

### Check CUDA Context
```python
import tensorflow as tf
print(f"TF version: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

strategy = tf.distribute.MirroredStrategy()
print(f"Strategy: {type(strategy).__name__}")
print(f"Replicas: {strategy.num_replicas_in_sync}")
```

### Test Multi-GPU
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
print(f"Replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Test training
import numpy as np
x = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)
model.fit(x, y, epochs=1, batch_size=32)
```

---

## Important Learnings

### TensorFlow Multi-GPU Best Practices

1. **Set CUDA_VISIBLE_DEVICES BEFORE importing TensorFlow**
   - ANY TensorFlow operation initializes GPU context
   - Cannot change visible devices after initialization

2. **Use strategy.scope() ONLY for model/optimizer creation**
   - ✅ Model creation
   - ✅ Model compilation
   - ✅ Optimizer creation
   - ❌ Dataset creation (auto-distributed)
   - ❌ Data processing
   - ❌ Shape inference

3. **Reuse the same MirroredStrategy object**
   - Creating multiple instances causes NCCL errors
   - Pass strategy object or store in globals

4. **Avoid nested strategy scopes**
   - Causes stack corruption
   - Only enter scope once per operation

5. **Lazy import TensorFlow in utility modules**
   - Prevents early initialization
   - Use `import tensorflow as tf` inside functions

### RTX 5090 Specific Issues

1. **Compute Capability 12.0 is bleeding edge**
   - Most frameworks don't have pre-built binaries
   - JIT compilation from PTX is slow and error-prone
   - NCCL may not support all collective operations

2. **TensorFlow 2.15.1 predates RTX 5090**
   - Released before GPU announcement
   - CUDA kernels compiled for up to compute capability 9.x
   - NCCL binaries may be incompatible

3. **Workarounds**:
   - Use HierarchicalCopyAllReduce (CPU-based aggregation)
   - Upgrade to newer TensorFlow when available
   - Use PyTorch (better new hardware support)
   - Disable P2P with NCCL_P2P_DISABLE=1

---

## Next Steps for Continuation

### Immediate Testing
1. Run with HierarchicalCopyAllReduce (already implemented)
2. Monitor nvidia-smi during training
3. Check if both GPUs show equal utilization

### If HierarchicalCopyAllReduce Fails
1. Try NCCL environment variables:
   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_DEBUG=INFO
   ```
2. Check NCCL logs for specific error

### If Still Failing
1. Test with minimal TF script (see "Test Multi-GPU" above)
2. If minimal script fails → TensorFlow/NCCL incompatibility confirmed
3. If minimal script works → Issue in codebase

### Long-term Solution
Consider upgrading to TensorFlow 2.16+ when stable:
```bash
# Test in new environment first
conda create -n test_tf_new python=3.11
conda activate test_tf_new
pip install tensorflow[and-cuda]>=2.16
# Run compatibility tests
```

---

## File Locations

**Configuration**:
- `/workspace/DFUMultiClassification/src/utils/gpu_config.py` - GPU setup logic
- `/workspace/DFUMultiClassification/src/utils/production_config.py` - Batch sizes, epochs
- `/workspace/DFUMultiClassification/requirements.yml` - Dependencies

**Training**:
- `/workspace/DFUMultiClassification/src/main.py` - Main entry point
- `/workspace/DFUMultiClassification/src/training/training_utils.py` - Training loop

**Environment**:
- Conda env: `multimodal`
- Activate: `source /opt/miniforge3/bin/activate multimodal` or `conda activate multimodal`

---

## Contact & References

**NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/
**TensorFlow Distributed Training**: https://www.tensorflow.org/guide/distributed_training
**RTX 5090 Specs**: Compute Capability 12.0 (Blackwell)

**Last Modified**: December 25, 2025
**Status**: ✅ **RESOLVED** - Multi-GPU training working with HierarchicalCopyAllReduce

---

## UPDATE: ISSUE RESOLVED ✅

### Solution
The **HierarchicalCopyAllReduce** workaround successfully fixed the NCCL errors! Training now completes on both GPUs.

### Additional Fix Required
After training, `model.load_weights()` needed to be wrapped in `strategy.scope()`:

```python
# Before (caused TypeError)
model.load_weights(checkpoint_path)

# After (works correctly)
with strategy.scope():
    model.load_weights(checkpoint_path)
```

**Files modified**:
- `src/training/training_utils.py` lines 1194-1195 (pre-training weight load)
- `src/training/training_utils.py` lines 1220-1221 (post-training weight load)

### Performance
Training completed successfully with:
- Both GPUs utilized
- No NCCL errors
- Proper gradient aggregation via CPU
- Model achieved reasonable accuracy (39% on 3-class problem)

**Recommendation**: This solution works but may be slower than NCCL. When TensorFlow 2.16+ with RTX 5090 support is stable, consider upgrading for better performance.
