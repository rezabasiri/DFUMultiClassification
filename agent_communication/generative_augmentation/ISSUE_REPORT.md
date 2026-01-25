# SDXL Generative Augmentation Issue Report

## Overview
Training is stuck at Fusion Stage 1 when using SDXL generative augmentation with TensorFlow MirroredStrategy (5x RTX 4090 GPUs).

## Current State
- **Process Status**: Running (PID 2311695, 117% CPU, 24GB RAM)
- **GPU Utilization**: 0% across all 5 GPUs
- **GPU Memory**: GPU 0 has 8.9GB used (SDXL model loaded), GPUs 1-4 have ~1.5GB each
- **Log File**: Stuck at 235 lines for several minutes

## What Works
1. SDXL generator loads successfully
2. SDXL generates images during pre-training phase (24 images for phases P and R)
3. Pre-training completes (kappa: 0.0000, expected for quick test)

## Where It Gets Stuck
The training hangs immediately after printing:
```
[GPU DEBUG] ========== FUSION STAGE 1 ==========
[GPU DEBUG] SDXL generative augmentation ACTIVE if enabled!
[GPU DEBUG] Expect HIGH GPU usage when SDXL generates images
[GPU DEBUG] =========================================
```

The next step should be `model.fit()` at `training_utils.py:1493` which should start producing epoch progress output.

## Root Cause Analysis
The issue appears to be a deadlock when TensorFlow MirroredStrategy tries to trace/distribute the augmentation function that contains `tf.py_function` calling SDXL.

### Key Code Section (generative_augmentation_sdxl.py:750-867)
```python
def create_enhanced_augmentation_fn(gen_manager, config):
    """
    Create TensorFlow augmentation function with generative augmentation

    IMPORTANT: This function is designed to work with MirroredStrategy.
    All generative augmentation logic is handled inside a single tf.py_function
    to avoid issues with tf.cond and lambda closures during graph tracing.
    """
    # Pre-check values at Python level
    gen_aug_enabled = config.modality_settings['depth_rgb']['generative_augmentations']['enabled']
    gen_aug_prob = config.generative_settings['prob']
    mix_ratio_min = config.generative_settings['mix_ratio_range'][0]
    mix_ratio_max = config.generative_settings['mix_ratio_range'][1]
    phases_lookup = tf.constant(['I', 'P', 'R'], dtype=tf.string)

    def _maybe_generate_impl(value_np, phase_bytes):
        """Implementation that runs in eager mode via tf.py_function."""
        try:
            # Convert EagerTensor to numpy array
            if hasattr(value_np, 'numpy'):
                value_array = value_np.numpy()
            else:
                value_array = np.array(value_np)

            # Probability check inside py_function (eager mode)
            if np.random.random() >= gen_aug_prob:
                return value_array

            # Phase handling...
            # Generate images via SDXL (PyTorch on GPU 0)
            generated = gen_manager.generate_images(
                'depth_rgb', phase,
                batch_size=batch_size,
                target_height=height,
                target_width=width
            )
            # Mix generated images with original batch...
            return result
        except Exception as e:
            print(f"Error in generative augmentation: {str(e)}")
            return value_array

    def apply_augmentation(features, label):
        for key, value in features.items():
            if should_apply_gen_aug:
                value = tf.py_function(
                    func=_maybe_generate_impl,
                    inp=[value, current_phase],
                    Tout=tf.float32
                )
                value.set_shape([None, None, None, 3])
        return output_features, label

    return apply_augmentation
```

## Previous Fixes Applied
1. **PyTorch 2.6 compatibility**: Added `weights_only=False` to `torch.load()`
2. **Removed @tf.function decorator**: Was causing retracing issues with MirroredStrategy
3. **Removed tf.cond**: Lambda closures were causing deadlocks during graph tracing
4. **EagerTensor conversion**: Fixed `.copy()` error by converting to numpy array at function start

## Potential Issues
1. **MirroredStrategy + tf.py_function**: When 5 replicas try to trace the py_function, they may all try to execute the SDXL generation on GPU 0 simultaneously
2. **No locking on generate_images()**: The SDXL generation isn't protected by a lock, only the initial loading is
3. **Graph compilation deadlock**: TensorFlow may be waiting indefinitely during dataset distribution

## Environment
- TensorFlow with MirroredStrategy
- 5x NVIDIA RTX 4090 (24GB each)
- PyTorch SDXL on GPU 0
- Python 3.11
- Quick test mode: 30% data, 3 epochs, 32x32 images, batch size 256

## Latest Log Output
```
2026-01-25 19:56:38,686 - INFO - QUICK TEST MODE: 30.0% data, 3 epochs, 32x32 images
2026-01-25 19:56:38,686 - INFO - This is for error checking only - results not production-ready

Skipping Baseline (no generative augmentation) (already completed)

================================================================================
TEST: With generative augmentation (depth_rgb)
================================================================================
Config updated: USE_GENERATIVE_AUGMENTATION=True
Modalities: metadata, depth_rgb, depth_map, thermal_map
Running: /venv/multimodal/bin/python src/main.py --data_percentage 30.0 --resume_mode fresh --device-mode multi

================================================================================
DEVICE CONFIGURATION (mode: multi)
================================================================================
Detected 5 GPU(s):
  GPU 0-4: NVIDIA GeForce RTX 4090 - 24.0GB (compute 8.9)

Using MirroredStrategy (5 GPUs) with NCCL
Batch size per replica: 51 (global batch size: 256, replicas: 5)

================================================================================
DFU MULTIMODAL CLASSIFICATION - PRODUCTION PIPELINE
================================================================================
Mode: search
Resume mode: fresh
Data percentage: 30.0%
Image size: 32x32
Batch size: 256
Max epochs: 3 (with early stopping)

FRESH START MODE: Deleting all checkpoints...

OUTLIER DETECTION AND REMOVAL
[1/1] depth_map_depth_rgb_metadata_thermal_map
  Total outliers: 410/2729 (15.0%)
  Cleaned dataset: 2319 samples

GENERATING 3-FOLD CROSS-VALIDATION SPLITS (PATIENT-LEVEL)
Class 0: 58 patients, Class 1: 95 patients, Class 2: 20 patients
Generated 3 folds

Fold 1/3
Using pre-computed fold 1 patient split
Using Scikit-learn RandomForestClassifier
Initializing GenerativeAugmentationManager with models from src/models/sdxl_checkpoints
✓ Found checkpoint: src/models/sdxl_checkpoints/checkpoint_best.pt
✓ Found config: /workspace/DFUMultiClassification/src/utils/full_sdxl_config.yaml
✓ SDXL paths configured (will load on first use)

Training metadata+depth_rgb+depth_map+thermal_map, fold 1/3
================================================================================
AUTOMATIC PRE-TRAINING: depth_rgb weights not found
  Training depth_rgb-only model first (same data split)...
================================================================================

[GPU DEBUG] PRE-TRAINING PHASE (depth_rgb)
[GPU DEBUG] GPU usage: LOW (small CNN, 64x64 images)
[GPU DEBUG] SDXL NOT LOADED YET - will load during FUSION training

============================================================
[GPU DEBUG] SDXL GENERATOR FIRST USE - LOADING NOW!
[GPU DEBUG] This will take ~30-60 seconds and use ~10GB GPU memory
============================================================

Loading SDXL generator (first use)...
✓ SDXL generator loaded successfully

============================================================
[GPU DEBUG] SDXL LOADING - Expect HIGH GPU memory usage soon!
============================================================
Loading SDXL checkpoint from src/models/sdxl_checkpoints/checkpoint_best.pt
Loading base model: stabilityai/stable-diffusion-xl-base-1.0
Loading trained UNet weights...
SDXL pipeline loaded successfully on cuda

[GPU DEBUG] SDXL GENERATING: 8 images for phase P at 32x32
[GPU DEBUG] Using 50 inference steps - Expect HIGH GPU utilization!
  Generated images: 8
[GPU DEBUG] SDXL GENERATING: 8 images for phase R at 32x32
[GPU DEBUG] Using 50 inference steps - Expect HIGH GPU utilization!
  Generated images: 16
[GPU DEBUG] SDXL GENERATING: 8 images for phase R at 32x32
[GPU DEBUG] Using 50 inference steps - Expect HIGH GPU utilization!
  Generated images: 24
  Pre-training completed! Best val kappa: 0.0000

================================================================================
No existing pretrained weights found

[GPU DEBUG] ========== FUSION STAGE 1 ==========
[GPU DEBUG] SDXL generative augmentation ACTIVE if enabled!
[GPU DEBUG] Expect HIGH GPU usage when SDXL generates images
[GPU DEBUG] =========================================

<--- STUCK HERE - No further output for several minutes --->
```

## Baseline Results (for comparison)
```json
{
  "baseline": {
    "kappa": 0.2867,
    "accuracy": 0.5753,
    "f1_macro": 0.484,
    "runtime_min": 158.48,
    "success": true
  }
}
```

## Questions for Investigation
1. How to properly use `tf.py_function` with PyTorch/CUDA operations inside MirroredStrategy?
2. Should the SDXL generation be moved to a separate process to avoid TensorFlow graph tracing issues?
3. Is there a way to make the augmentation function run only on the main replica?
4. Should we add a threading lock around the SDXL generation to serialize GPU access?
