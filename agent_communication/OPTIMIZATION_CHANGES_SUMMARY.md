# Optimization Changes Summary

**Date:** 2026-02-10
**Focus:** Bottleneck #1 (CPU-based image loading) + Bottleneck #2 (CPU-based augmentation) + Time Debugging Instrumentation

---

## Changes Made

### 1. **Bottleneck #1: CPU-Based Image Loading → GPU-Accelerated (CRITICAL)** ✅

**File:** `src/data/dataset_utils.py`
**Lines:** 234-337 (replaced `process_single_sample` function)

#### What Changed:
- **REMOVED:** PIL-based image loading via `tf.py_function` (CPU-only, Python GIL-locked)
- **ADDED:** Native TensorFlow GPU-accelerated image loading using `tf.io.read_file()` + `tf.io.decode_jpeg()`

#### Key Improvements:
1. **GPU JPEG Decoding:** Uses `tf.io.decode_jpeg()` which can run on GPU
2. **Graph Execution:** Entire pipeline stays in TensorFlow graph (no Python GIL)
3. **Parallelization:** TensorFlow can parallelize operations across GPU cores
4. **Conditional Logic:** Uses `tf.case()` for modality-specific folder selection
5. **Bounding Box Adjustments:** Implemented `adjust_depth_map_bb()` and `adjust_thermal_map_bb()` in pure TensorFlow
6. **Aspect Ratio Preservation:** Maintains aspect ratio with padding using `tf.image` operations
7. **Normalization:** Modality-specific normalization (RGB: /255, MAP: /max or /255)

#### Expected Impact:
- **2-3x speedup** on image loading
- GPU compute utilization should increase from ~0-30% to ~40-60% (will improve further with other optimizations)
- Training step time should decrease from 1.2-1.7 sec/step to ~0.6-0.9 sec/step

#### Code Structure:
```python
def process_single_sample(filename, bb_coords, modality_name):
    # Build file path with tf.case (GPU-compatible)
    img_path = tf.case([...], default=..., exclusive=True)

    # GPU-accelerated JPEG decode
    img_raw = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_raw, channels=3)

    # Extract and adjust bounding box (all TensorFlow ops)
    xmin, ymin, xmax, ymax = ...

    # Crop, resize with aspect ratio, pad
    cropped_img = tf.image.crop_to_bounding_box(...)
    resized_img = tf.image.resize(..., method='lanczos3')
    padded_img = tf.pad(resized_img, ...)

    # Normalize based on modality
    normalized_img = tf.case([...], default=...)

    return normalized_img
```

---

### 2. **Time Debugging Instrumentation** ✅

Added `[TIME_DEBUG]` markers at all critical points for performance profiling. All prints use `flush=True` for real-time logging.

#### 2.1 Dataset Creation Timing
**File:** `src/data/dataset_utils.py`
**Lines:** 215-217, 373-374, 397-398, 412-413, 425-427

```python
# START
print(f"[TIME_DEBUG] create_cached_dataset START (fold={fold_id}, training={is_training})", flush=True)

# After image loading map
print(f"[TIME_DEBUG] Image loading map completed: {t_after_map - t_start:.2f}s", flush=True)

# After cache setup
print(f"[TIME_DEBUG] Cache setup completed: {t_after_cache - t_after_map:.2f}s", flush=True)

# After augmentation setup (if training)
print(f"[TIME_DEBUG] Augmentation setup: {t_after_aug - t_after_cache:.2f}s", flush=True)

# COMPLETE
print(f"[TIME_DEBUG] create_cached_dataset COMPLETE: {t_end - t_start:.2f}s total", flush=True)
```

#### 2.2 RandomForest Training Timing
**File:** `src/data/dataset_utils.py`
**Lines:** 1195-1197, 1258-1264, 1292-1298, 1333-1336

```python
# START
print(f"[TIME_DEBUG] RandomForest training START", flush=True)

# After model creation
print(f"[TIME_DEBUG] RF models created (tfdf/sklearn)", flush=True)

# After RF model 1 trained
print(f"[TIME_DEBUG] RF model 1 trained: {t_rf1_done - t_rf_start:.2f}s", flush=True)

# After RF model 2 trained
print(f"[TIME_DEBUG] RF model 2 trained: {t_rf2_done - t_rf1_done:.2f}s", flush=True)

# After predictions
print(f"[TIME_DEBUG] RF predictions complete: {t_rf_end - t_rf2_done:.2f}s", flush=True)

# TOTAL
print(f"[TIME_DEBUG] RandomForest TOTAL: {t_rf_end - t_rf_start:.2f}s", flush=True)
```

#### 2.3 Pre-training Phase Timing
**File:** `src/training/training_utils.py`
**Lines:** 1287-1289, 1313-1314, 1327-1328, 1380-1395, 1430-1431

```python
# START
print(f"[TIME_DEBUG] Pre-training phase START", flush=True)

# After model compile
print(f"[TIME_DEBUG] Pre-train model compiled: {t_after_compile - t_pretrain_start:.2f}s", flush=True)

# After dataset distribution
print(f"[TIME_DEBUG] Datasets distributed to GPUs: {t_after_dist - t_after_compile:.2f}s", flush=True)

# Before model.fit
print(f"[TIME_DEBUG] Starting pre-train model.fit() with {max_epochs} max epochs", flush=True)

# After model.fit
print(f"[TIME_DEBUG] Pre-train model.fit() completed: {t_fit_end - t_fit_start:.2f}s ({(t_fit_end - t_fit_start)/60:.1f} min)", flush=True)

# TOTAL
print(f"[TIME_DEBUG] Pre-training TOTAL: {t_pretrain_end - t_pretrain_start:.2f}s ({(t_pretrain_end - t_pretrain_start)/60:.1f} min)", flush=True)
```

#### 2.4 Main Training Stages Timing
**File:** `src/training/training_utils.py`
**Lines:** 1598-1600, 1630-1632, 1660-1662, 1691-1693

```python
# Stage 1 START
print(f"[TIME_DEBUG] Stage 1 training START (max_epochs={STAGE1_EPOCHS})", flush=True)

# Stage 1 COMPLETE
print(f"[TIME_DEBUG] Stage 1 training COMPLETE: {t_stage1_end - t_stage1_start:.2f}s ({(t_stage1_end - t_stage1_start)/60:.1f} min)", flush=True)

# Stage 2 START
print(f"[TIME_DEBUG] Stage 2 training START (max_epochs=100)", flush=True)

# Stage 2 COMPLETE
print(f"[TIME_DEBUG] Stage 2 training COMPLETE: {t_stage2_end - t_stage2_start:.2f}s ({(t_stage2_end - t_stage2_start)/60:.1f} min)", flush=True)
```

#### 2.5 Per-Epoch Timing
**File:** `src/training/training_utils.py`
**Lines:** 100-145 (modified `PeriodicEpochPrintCallback` class)

```python
class PeriodicEpochPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_interval=50, total_epochs=20):
        super().__init__()
        self.print_interval = print_interval if print_interval > 0 else 1
        self.total_epochs = total_epochs
        self.epoch_start_time = None  # ADDED

    def on_epoch_begin(self, epoch, logs=None):  # ADDED
        import time
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        import time
        epoch_time = time.time() - self.epoch_start_time  # ADDED

        # Print with [TIME_DEBUG] prefix and epoch time
        metrics_str = f"[TIME_DEBUG] Epoch {epoch_num}/{self.total_epochs} - {epoch_time:.1f}s"
        # ... rest of metrics ...
        print(metrics_str, flush=True)  # ADDED flush=True
```

#### 2.6 Image Loading Performance (Sampling)
**File:** `src/data/image_processing.py`
**Lines:** 320-325, 337-339, 443-446

**NOTE:** This is the OLD PIL-based loading function. It's still used as a fallback but should NOT be called after our optimization (process_single_sample now uses tf.io). Timing is included for verification.

```python
def load_and_preprocess_image(filepath, bb_data, modality, target_size=(224, 224), augment=False):
    import time
    import random

    # Sample timing every 1% of images
    should_time = random.random() < 0.01

    if should_time:
        t_start = time.time()

    # ... PIL loading ...

    if should_time:
        t_after_load = time.time()
        print(f"[TIME_DEBUG] Image load (PIL): {(t_after_load - t_start)*1000:.1f}ms", flush=True)

    # ... preprocessing ...

    if should_time:
        t_end = time.time()
        print(f"[TIME_DEBUG] Image preprocess total: {(t_end - t_start)*1000:.1f}ms", flush=True)
```

#### 2.7 Augmentation Performance (Sampling) - NEW
**File:** `src/data/generative_augmentation_v3.py`
**Lines:** 688-691, 831-834

```python
# Track augmentation timing (sample every 100 batches)
_aug_timing_counter = [0]
_aug_timing_interval = 100

def apply_augmentation(features, label):
    def augment_batch(features_dict, label_tensor):
        # Timing: Sample every N batches
        should_time = (_aug_timing_counter[0] % _aug_timing_interval == 0)
        if should_time:
            t_aug_start = time.time()

        # ... augmentation logic ...

        # Timing: Print if this batch was sampled
        if should_time:
            t_aug_end = time.time()
            print(f"[TIME_DEBUG] Augmentation batch (GPU): {(t_aug_end - t_aug_start)*1000:.1f}ms", flush=True)

        _aug_timing_counter[0] += 1
```

---

## Files Modified - Summary

### Bottleneck #1: GPU Image Loading
| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/data/dataset_utils.py` | 234-337 | **CRITICAL:** Replaced PIL with tf.io GPU-accelerated loading |

### Bottleneck #2: GPU Augmentation
| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/data/generative_augmentation_v3.py` | 269-285 | **CRITICAL:** Removed CPU device placement |
| `src/data/generative_augmentation_v3.py` | 688-691, 831-834 | Added augmentation timing instrumentation |
| `src/data/generative_augmentation_v2.py` | 234-249 | Removed CPU device placement (backward compatibility) |

### Time Debugging Instrumentation
| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/data/dataset_utils.py` | 215-217, 373-374, 397-398, 412-413, 425-427 | Dataset creation timing |
| `src/data/dataset_utils.py` | 1195-1197, 1258-1264, 1292-1298, 1333-1336 | RandomForest timing |
| `src/training/training_utils.py` | 100-145 | Per-epoch timing in callback |
| `src/training/training_utils.py` | 1287-1289, 1313-1314, 1327-1328, 1380-1395, 1430-1431 | Pre-training timing |
| `src/training/training_utils.py` | 1598-1600, 1630-1632, 1660-1662, 1691-1693 | Stage 1/2 timing |
| `src/data/image_processing.py` | 320-325, 337-339, 443-446 | PIL image loading timing (fallback only) |
| `src/data/generative_augmentation_v3.py` | 688-691, 831-834 | Augmentation batch timing |

---

## How to Test

### Quick Test (Single Fold):
```bash
python src/main.py --data_percentage 30 --device-mode multi --verbosity 2 --fold 1
```
- **Before (baseline):** ~148 minutes (442 min / 3 folds)
- **After Bottleneck #1:** ~50-75 minutes (2-3x faster)
- **After Bottlenecks #1 + #2:** ~30-50 minutes (3-5x faster)

### Full Test (3 Folds):
```bash
python agent_communication/generative_augmentation/test_generative_aug.py
```
- **Before (baseline):** 442 minutes
- **After Bottleneck #1:** ~150-220 minutes (2-3x faster)
- **After Bottlenecks #1 + #2:** ~90-150 minutes (3-5x faster)

### What to Look For:
1. **GPU Utilization:** Check `nvidia-smi` during training
   - Before (baseline): ~0-30% GPU compute
   - After Bottleneck #1: ~40-60% GPU compute
   - After Bottlenecks #1 + #2: ~60-80% GPU compute (will improve to 70-95% with remaining optimizations)

2. **Step Time:** Look at epoch timing in logs
   - Before (baseline): 1.2-1.7 sec/step
   - After Bottleneck #1: ~0.6-0.9 sec/step
   - After Bottlenecks #1 + #2: ~0.4-0.6 sec/step

3. **[TIME_DEBUG] Logs:** Parse logs to see breakdown:
   ```bash
   grep "[TIME_DEBUG]" results/logs/training.log
   ```

   Look for:
   - Image loading timing (should be much faster with GPU)
   - Augmentation batch timing (every 100 batches, should show GPU speedup)
   - Per-epoch timing breakdown

---

## Removing Time Debug Instrumentation

After optimization is complete and verified, remove all `[TIME_DEBUG]` prints:

```bash
# Find all TIME_DEBUG lines
grep -r "\[TIME_DEBUG\]" src/

# Remove manually or use sed (be careful):
# Backup first!
find src/ -name "*.py" -type f -exec sed -i.bak '/\[TIME_DEBUG\]/d' {} \;
```

**Files to clean:**
- `src/data/dataset_utils.py`
- `src/training/training_utils.py`
- `src/data/image_processing.py`

---

---

## Bottleneck #2: CPU-Based Data Augmentation → GPU Augmentation (CRITICAL) ✅

**Date:** 2026-02-10

### Problem Identified

**Location:** `src/data/generative_augmentation_v3.py:272` and `src/data/generative_augmentation_v2.py:237`

Augmentations were **explicitly forced to run on CPU** with `with tf.device('/CPU:0'):` to avoid deterministic GPU issues with `tf.image.adjust_contrast()`. This caused the entire augmentation pipeline to be CPU-bound, creating a critical bottleneck where the CPU preprocesses batches while GPUs sit idle.

**Impact:** 2-3x slowdown during training

### Changes Made

#### File: `src/data/generative_augmentation_v3.py`

**Lines 269-285:** Removed CPU device placement in `augment_image()` function
```python
# BEFORE (CPU-bound):
with tf.device('/CPU:0'):
    if modality in ['depth_rgb', 'thermal_rgb']:
        augmented = tf.map_fn(...)
    else:
        augmented = tf.map_fn(...)

# AFTER (GPU-accelerated):
if modality in ['depth_rgb', 'thermal_rgb']:
    augmented = tf.map_fn(...)
else:
    augmented = tf.map_fn(...)
```

**Lines 672-695:** Added timing instrumentation for augmentation
```python
# Track augmentation timing (sample every 100 batches)
import time
_aug_timing_counter = [0]
_aug_timing_interval = 100

def apply_augmentation(features, label):
    def augment_batch(features_dict, label_tensor):
        # Timing: Sample every N batches
        should_time = (_aug_timing_counter[0] % _aug_timing_interval == 0)
        if should_time:
            t_aug_start = time.time()
```

**Lines 828-835:** Added timing print statement
```python
# Timing: Print if this batch was sampled
if should_time:
    t_aug_end = time.time()
    print(f"[TIME_DEBUG] Augmentation batch (GPU): {(t_aug_end - t_aug_start)*1000:.1f}ms", flush=True)

_aug_timing_counter[0] += 1
```

#### File: `src/data/generative_augmentation_v2.py`

**Lines 234-249:** Removed CPU device placement (same fix as v3 for backward compatibility)

### Technical Details

**GPU-Compatible Operations:** All `tf.image` operations used are GPU-compatible:
- `tf.image.random_brightness()` - GPU accelerated
- `tf.image.random_contrast()` - GPU accelerated
- `tf.image.random_saturation()` - GPU accelerated
- `tf.random.normal()` (for Gaussian noise) - GPU accelerated
- `tf.clip_by_value()` - GPU accelerated

**Determinism Note:** If deterministic behavior is required (e.g., for reproducibility), set the `TF_DETERMINISTIC_OPS=1` environment variable. This was the original reason for CPU placement, but it's better to let users opt-in to determinism rather than force all augmentations to CPU.

### Expected Impact

- **2-3x speedup** on augmentation portion of pipeline
- GPU compute utilization should increase from ~40-60% (after Bottleneck #1) to ~60-80%
- Training step time should decrease further from ~0.6-0.9 sec/step to ~0.4-0.6 sec/step
- **Combined with Bottleneck #1:** Total speedup of 3-5x expected

### Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/data/generative_augmentation_v3.py` | 269-285 | Removed CPU device placement in augment_image() |
| `src/data/generative_augmentation_v3.py` | 672-695 | Added timing counter and instrumentation setup |
| `src/data/generative_augmentation_v3.py` | 828-835 | Added timing print statement |
| `src/data/generative_augmentation_v2.py` | 234-249 | Removed CPU device placement (backward compatibility) |

---

## Next Steps (NOT Done Yet)

From `performance_investigation.md`, these optimizations are **NOT** included yet:

- ✅ **Bottleneck #1:** CPU-based image loading → GPU-accelerated (DONE)
- ✅ **Bottleneck #2:** CPU-based data augmentation → GPU augmentation (DONE)
- ❌ **Bottleneck #3:** Small batch size (32) → Increase to 128+
- ❌ **Bottleneck #4:** RandomForest caching
- ❌ **Bottleneck #5:** Pre-training cache fix

**Expected Final Speedup:** 4-8x (442 min → 55-110 min) after ALL optimizations
**Current Progress:** Bottlenecks #1 + #2 should give 3-5x speedup (442 min → ~90-150 min)

---

## Important Notes

1. **Code Changes:** Bottleneck #1 (GPU image loading), Bottleneck #2 (GPU augmentation), and timing instrumentation implemented
2. **Backward Compatibility:**
   - Old PIL function (`load_and_preprocess_image`) is still present for fallback/debugging
   - Both v3 and v2 augmentation files fixed for compatibility
3. **Parameter Consistency:** All variable/parameter names maintained exactly as in original code
4. **Modality Logic:** Correctly handles all 4 modalities (depth_rgb, depth_map, thermal_rgb, thermal_map)
5. **Bounding Box Logic:** Replicates exact adjustment logic from `adjust_bounding_box()` and thermal +30 padding
6. **Aspect Ratio:** Maintains aspect ratio with black padding (same as original PIL version)
7. **GPU Placement:** Augmentations now run on GPU by default; use `TF_DETERMINISTIC_OPS=1` if reproducibility is critical

---

## Verification Checklist

After running test:

### Bottleneck #1 (GPU Image Loading)
- [ ] GPU compute % increased from ~0-30% to ~40-60%
- [ ] Step time decreased from 1.2-1.7s to ~0.6-0.9s
- [ ] No errors in image loading (check for default black images)
- [ ] All 4 modalities load correctly (depth_rgb, depth_map, thermal_rgb, thermal_map)

### Bottleneck #2 (GPU Augmentation)
- [ ] GPU compute % increased from ~40-60% to ~60-80%
- [ ] Step time decreased further from ~0.6-0.9s to ~0.4-0.6s
- [ ] `[TIME_DEBUG]` logs show augmentation batch timing (sampled every 100 batches)
- [ ] Augmentations still work correctly (check saved sample images in results/visualizations/)

### General
- [ ] No accuracy degradation (Kappa should be similar ±0.02)
- [ ] `[TIME_DEBUG]` logs show breakdown of time spent in all phases
- [ ] Combined speedup of 3-5x (442 min → ~90-150 min expected for full run)

---

**Generated:** 2026-02-10
**By:** Claude Sonnet 4.5
