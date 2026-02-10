# Performance Investigation: Test 1 (Baseline) - 442 Minutes

**Date:** 2026-02-10
**Test:** Baseline (no SDXL generative augmentation)
**Total Time:** 442.6 minutes (7h 22m)
**Expected Time:** ~55-110 minutes (4-8x speedup possible)

---

## Key Finding: GPU-Starved by CPU Data Pipeline

**Observed Performance:**
- Training: 1.2-1.7 seconds/step (should be 0.05-0.2 sec/step)
- **8-16x slower than expected**
- GPU compute utilization: ~0-30% (GPU waiting for CPU)
- CPU utilization: 7% (underutilized despite bottleneck)

**Root Cause:**
- Image loading happens on CPU (PIL-based, single-threaded)
- Data augmentation runs on CPU during training
- Small batch size (32) doesn't saturate 2 GPUs
- GPU is memory-bound but compute-starved

---

## Time Breakdown (Test 1)

```
Total: 442 minutes
├── Pre-training (3 folds): 223 min (50.5%)
│   ├── Fold 1: 114 min (99 epochs × 69 sec/epoch)
│   ├── Fold 2: 66 min (50 epochs)
│   └── Fold 3: 43 min (36 epochs)
│
├── Stage 1 (3 folds): 112 min (25.3%)
│   ├── Fold 1: 19 min (11 epochs × 104 sec/epoch)
│   ├── Fold 2: 38 min (20 epochs)
│   └── Fold 3: 55 min (33 epochs)
│
├── Stage 2 (3 folds): 83 min (18.8%)
│   ├── Fold 1: 26 min (11 epochs × 142 sec/epoch)
│   ├── Fold 2: 30 min (11 epochs)
│   └── Fold 3: 27 min (11 epochs)
│
└── Overhead: ~17 min (3.8%)
    ├── RandomForest training: ~9 min
    ├── Dataset preparation: ~5 min
    └── Gating network: ~3 min
```

**Calculations:**
- Dataset: 2700 samples, batch size 32 = 84 steps/epoch
- Current: 1.6 sec/step × 84 = 134 sec/epoch
- Target: 0.1 sec/step × 84 = 8.4 sec/epoch
- **16x speedup possible on training loop**

---

## Bottleneck #1: CPU-Based Image Loading (CRITICAL)

**Location:** `src/data/dataset_utils.py:234-274`

**Problem:**
```python
def _process_image(filename_tensor, bb_coords_tensor, modality_tensor):
    # PIL.Image loading - CPU-only, single-threaded
    img_tensor = load_and_preprocess_image(img_path, ...)
    return img_array

processed_image = tf.py_function(
    _process_image,  # Forces Python execution (GIL-locked)
    [filename, bb_coords, modality_name],
    tf.float32
)
```

**Why It's Slow:**
- `tf.py_function` breaks TensorFlow graph execution
- PIL image loading is CPU-only and synchronous
- Python GIL prevents parallelization
- GPU waits idle while CPU loads images

**Impact:** 2-3x slowdown

**Options to Fix:**

1. **Use `tf.io` native functions (RECOMMENDED)**
   - File: `src/data/dataset_utils.py:234-274`
   - Replace PIL with `tf.io.read_file()` + `tf.io.decode_jpeg()`
   - Enable GPU-accelerated JPEG decoding
   - Difficulty: Medium

2. **Use `tf.data.Dataset.interleave()` for parallel loading**
   - Add parallel file reading with `cycle_length` parameter
   - Difficulty: Easy

3. **Pre-cache decoded images to disk**
   - Save preprocessed images as TFRecord files
   - Trade disk space for speed
   - Difficulty: Medium

**Action Plan:**
```python
# OPTION 1: Replace in src/data/dataset_utils.py:234-274
def process_single_sample_native(filename, bb_coords, modality_name):
    """GPU-accelerated image loading using tf.io"""
    # Build file path
    base_folder = tf.case([
        (tf.equal(modality_name, 'depth_rgb'), lambda: image_folder),
        (tf.equal(modality_name, 'depth_map'), lambda: depth_folder),
        # ... etc
    ])
    img_path = tf.strings.join([base_folder, '/', filename])

    # GPU-accelerated decode
    img_raw = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_raw, channels=3)

    # Crop with bounding box
    img = tf.image.crop_to_bounding_box(img, ...)

    # Resize
    img = tf.image.resize(img, [image_size, image_size])

    return img / 255.0
```

---

## Bottleneck #2: CPU-Based Data Augmentation (CRITICAL)

**Location:** `src/data/dataset_utils.py:404-411`

**Problem:**
```python
if augmentation_fn:
    dataset = dataset.map(
        augmentation_fn,  # Runs on CPU during training
        num_parallel_calls=tf.data.AUTOTUNE,
    )
```

**Referenced in:** `src/data/generative_augmentation_v3.py:create_enhanced_augmentation_fn()`

**Why It's Slow:**
- Augmentation (brightness, contrast, flip, rotate) happens synchronously during `model.fit()`
- CPU preprocesses batch while GPU waits
- Even with `AUTOTUNE`, CPU can't keep up with GPU

**Impact:** 2-3x slowdown

**Options to Fix:**

1. **Move augmentation to GPU using `tf.image` ops (RECOMMENDED)**
   - File: `src/data/generative_augmentation_v3.py` or equivalent
   - Replace numpy/PIL operations with `tf.image` operations
   - Augmentation runs on GPU during forward pass
   - Difficulty: Easy-Medium

2. **Use `tf.data.Dataset.map()` with GPU device placement**
   - Wrap augmentation in `with tf.device('/GPU:0'):`
   - Difficulty: Easy

3. **Pre-augment images offline**
   - Generate augmented versions before training
   - Trade disk space for speed
   - Difficulty: Easy but wasteful

**Action Plan:**
```python
# OPTION 1: Modify augmentation function to use tf.image
def create_gpu_augmentation_fn(prob=0.3):
    """GPU-accelerated augmentation"""
    def augment(inputs, labels):
        for modality in inputs.keys():
            if 'input' in modality:
                img = inputs[modality]

                # All operations are GPU-compatible
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
                img = tf.image.random_flip_left_right(img)
                img = tf.clip_by_value(img, 0.0, 1.0)

                inputs[modality] = img
        return inputs, labels

    return augment
```

---

## Bottleneck #3: Small Batch Size on Multi-GPU (MAJOR)

**Location:** `src/utils/production_config.py:29`

**Problem:**
```python
GLOBAL_BATCH_SIZE = 32  # Too small for 2 GPUs
```

**GPU Status:**
```
GPU 0: 9.3GB / 24GB used (38% memory, ~0% compute)
GPU 1: 0.2GB / 24GB used (1% memory, ~0% compute)
```

**Why It's Slow:**
- Batch size 32 ÷ 2 GPUs = 16 samples per GPU
- Too small to saturate GPU compute units
- More time spent on synchronization overhead than actual compute
- GPU 1 is practically idle

**Impact:** 1.5-2x slowdown

**Options to Fix:**

1. **Increase GLOBAL_BATCH_SIZE to 128-256 (RECOMMENDED)**
   - File: `src/utils/production_config.py:29`
   - Change: `GLOBAL_BATCH_SIZE = 128`
   - Each GPU gets 64 samples (better utilization)
   - Difficulty: Easy
   - **Risk:** May need to reduce learning rate accordingly

2. **Use gradient accumulation**
   - Simulate larger batch size without memory increase
   - Difficulty: Medium

3. **Single GPU with larger batch**
   - Only use GPU 0 with batch size 64-128
   - Difficulty: Easy

**Action Plan:**
```python
# OPTION 1: Modify src/utils/production_config.py:29
GLOBAL_BATCH_SIZE = 128  # Was 32

# May need to adjust learning rate in src/training/training_utils.py:1306
optimizer=Adam(learning_rate=1e-4 * (128/32), clipnorm=1.0)  # Scale LR
```

---

## Bottleneck #4: RandomForest Metadata Training (MODERATE)

**Location:** `src/data/dataset_utils.py:1185-1213`

**Problem:**
```python
rf_model1 = RandomForestClassifier(
    n_estimators=646,  # 646 trees!
    n_jobs=-1          # CPU-only (scikit-learn has no GPU version)
)
rf_model2 = RandomForestClassifier(
    n_estimators=646,  # Another 646 trees!
    n_jobs=-1
)
rf_model1.fit(X, y_bin1)  # ~3-5 minutes on CPU
rf_model2.fit(X, y_bin2)  # ~3-5 minutes on CPU
```

**Why It's Slow:**
- 1,292 decision trees total (2 × 646)
- Scikit-learn is CPU-only
- Runs once per fold (3 times total)

**Impact:** ~9-15 minutes total (not huge, but CPU-bound)

**Options to Fix:**

1. **Cache RandomForest predictions (RECOMMENDED)**
   - File: `src/data/dataset_utils.py:1212-1234`
   - Save RF predictions to disk after first training
   - Load cached predictions in subsequent runs
   - Difficulty: Easy

2. **Use GPU-accelerated RandomForest**
   - Replace scikit-learn with cuML (RAPIDS)
   - Requires CUDA-compatible GPU libraries
   - Difficulty: Hard (dependency management)

3. **Reduce n_estimators**
   - Change 646 → 200 (test if performance drops)
   - Difficulty: Easy

4. **Use TensorFlow Decision Forests (already attempted)**
   - Code tries `tensorflow_decision_forests` first (line 1170-1179)
   - Falls back to scikit-learn on ImportError
   - Install tfdf to use GPU version
   - Difficulty: Easy

**Action Plan:**
```python
# OPTION 1: Add caching in src/data/dataset_utils.py:1212-1234
import pickle
cache_file = f"results/rf_cache/rf_predictions_fold{run}_data{DATA_PERCENTAGE}.pkl"

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        prob1, prob2, prob3 = pickle.load(f)
    vprint(f"Loaded cached RF predictions from {cache_file}")
else:
    # Train and predict
    rf_model1.fit(X, y_bin1)
    rf_model2.fit(X, y_bin2)
    prob1 = rf_model1.predict_proba(dataset)[:, 1]
    prob2 = rf_model2.predict_proba(dataset)[:, 1]
    prob3 = 1 - prob1 * prob2

    # Save cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((prob1, prob2, prob3), f)
    vprint(f"Saved RF predictions to cache: {cache_file}")
```

---

## Bottleneck #5: Pre-training Cache Not Working (MODERATE)

**Location:** `src/training/training_utils.py:67-78, 1274-1283, 1409-1415`

**Problem:**
- Cache exists but never loads: `results/pretrain_cache/pretrain_depth_rgb_fold1_*.ckpt`
- Log shows "depth_rgb weights not found" for every fold
- Each fold re-trains from scratch (43-114 min per fold)

**Why It Fails:**
- Cache save works (line 1409-1415)
- Cache load probably fails silently (exception caught but not logged)
- May be TensorFlow checkpoint format mismatch

**Impact:** If interrupted/resumed, wastes 1-2 hours re-training

**Options to Fix:**

1. **Add verbose logging to cache load (RECOMMENDED)**
   - File: `src/training/training_utils.py:1274-1283`
   - Print exception details when cache load fails
   - Difficulty: Easy

2. **Fix checkpoint format compatibility**
   - Ensure save/load use same TensorFlow checkpoint API
   - May need to use `tf.train.Checkpoint` instead of `model.save_weights`
   - Difficulty: Medium

**Action Plan:**
```python
# OPTION 1: Add logging in src/training/training_utils.py:1274-1283
try:
    pretrain_cache_path = _get_pretrain_cache_path(image_modality, run+1)
    vprint(f"  Checking for cached pre-trained weights: {pretrain_cache_path}", level=2)

    if os.path.exists(pretrain_cache_path + '.index'):  # TF checkpoint
        vprint(f"  Cache file found, attempting to load...", level=2)
        pretrain_model = create_multimodal_model(input_shapes, [image_modality], None)
        pretrain_model.load_weights(pretrain_cache_path)
        vprint(f"  ✓ Loaded from cache - skipping ~17min pre-training!", level=1)
        fusion_use_pretrained = True
    else:
        vprint(f"  Cache not found (looking for {pretrain_cache_path}.index)", level=2)
except Exception as e:
    vprint(f"  Cache load failed: {type(e).__name__}: {e}", level=1)
    import traceback
    vprint(f"  Traceback: {traceback.format_exc()}", level=2)
```

---

## Time Debugging Instrumentation

Add these timing prints to narrow down bottlenecks. Mark with `[TIME_DEBUG]` prefix for easy removal.

### 1. Dataset Creation Timing

**File:** `src/data/dataset_utils.py:213-421`

```python
import time

def create_cached_dataset(best_matching_df, selected_modalities, batch_size,
                         is_training=True, cache_dir=None, augmentation_fn=None, image_size=128, fold_id=0):
    """Create a cached TF dataset..."""

    print(f"[TIME_DEBUG] create_cached_dataset START (fold={fold_id}, training={is_training})", flush=True)
    t_start = time.time()

    # ... existing code for dataset creation ...

    # After line 366 (after dataset.map for image loading)
    t_after_map = time.time()
    print(f"[TIME_DEBUG] Image loading map completed: {t_after_map - t_start:.2f}s", flush=True)

    # After line 391 (after .cache())
    t_after_cache = time.time()
    print(f"[TIME_DEBUG] Cache setup completed: {t_after_cache - t_after_map:.2f}s", flush=True)

    # After line 411 (after augmentation)
    if is_training and augmentation_fn:
        t_after_aug = time.time()
        print(f"[TIME_DEBUG] Augmentation setup: {t_after_aug - t_after_cache:.2f}s", flush=True)

    # At return (line 421)
    t_end = time.time()
    print(f"[TIME_DEBUG] create_cached_dataset COMPLETE: {t_end - t_start:.2f}s total", flush=True)

    return dataset, pre_aug_dataset, steps
```

### 2. RandomForest Training Timing

**File:** `src/data/dataset_utils.py:1180-1214`

```python
# Before line 1181 (before vprint "Using Scikit-learn")
print(f"[TIME_DEBUG] RandomForest training START", flush=True)
t_rf_start = time.time()

# After line 1194 (after creating rf_model1)
print(f"[TIME_DEBUG] RF models created", flush=True)

# After line 1212 (after rf_model1.fit)
t_rf1_done = time.time()
print(f"[TIME_DEBUG] RF model 1 trained: {t_rf1_done - t_rf_start:.2f}s", flush=True)

# After line 1213 (after rf_model2.fit)
t_rf2_done = time.time()
print(f"[TIME_DEBUG] RF model 2 trained: {t_rf2_done - t_rf1_done:.2f}s", flush=True)

# After prediction (line 1234)
t_rf_end = time.time()
print(f"[TIME_DEBUG] RF predictions complete: {t_rf_end - t_rf2_done:.2f}s", flush=True)
print(f"[TIME_DEBUG] RandomForest TOTAL: {t_rf_end - t_rf_start:.2f}s", flush=True)
```

### 3. Pre-training Phase Timing

**File:** `src/training/training_utils.py:1285-1420`

```python
# Before line 1287 (before "AUTOMATIC PRE-TRAINING")
print(f"[TIME_DEBUG] Pre-training phase START", flush=True)
t_pretrain_start = time.time()

# After line 1310 (after model compile)
t_after_compile = time.time()
print(f"[TIME_DEBUG] Pre-train model compiled: {t_after_compile - t_pretrain_start:.2f}s", flush=True)

# After line 1324 (after dataset distribution)
t_after_dist = time.time()
print(f"[TIME_DEBUG] Datasets distributed to GPUs: {t_after_dist - t_after_compile:.2f}s", flush=True)

# Before line 1371 (before pretrain_model.fit)
print(f"[TIME_DEBUG] Starting pre-train model.fit() with {max_epochs} max epochs", flush=True)
t_fit_start = time.time()

# After line 1379 (after pretrain_model.fit)
t_fit_end = time.time()
print(f"[TIME_DEBUG] Pre-train model.fit() completed: {t_fit_end - t_fit_start:.2f}s ({(t_fit_end - t_fit_start)/60:.1f} min)", flush=True)

# After line 1420 (after weight transfer complete)
t_pretrain_end = time.time()
print(f"[TIME_DEBUG] Pre-training TOTAL: {t_pretrain_end - t_pretrain_start:.2f}s ({(t_pretrain_end - t_pretrain_start)/60:.1f} min)", flush=True)
```

### 4. Main Training Stages Timing

**File:** `src/training/training_utils.py` (around fusion training model.fit calls)

Search for `model.fit(` calls and add before/after:

```python
# Before each model.fit() call
print(f"[TIME_DEBUG] Stage X training START (max_epochs={max_epochs})", flush=True)
t_stage_start = time.time()

# model.fit() call here

# After each model.fit() call
t_stage_end = time.time()
print(f"[TIME_DEBUG] Stage X training COMPLETE: {t_stage_end - t_stage_start:.2f}s ({(t_stage_end - t_stage_start)/60:.1f} min)", flush=True)
```

### 5. Per-Epoch Timing (if verbose enough)

**File:** `src/training/training_utils.py:99-137` (PeriodicEpochPrintCallback)

```python
class PeriodicEpochPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_interval=50, total_epochs=20):
        super().__init__()
        self.print_interval = print_interval if print_interval > 0 else 1
        self.total_epochs = total_epochs
        self.epoch_start_time = None  # ADD THIS

    def on_epoch_begin(self, epoch, logs=None):  # ADD THIS METHOD
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1

        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        should_print = (
            epoch_num == 1 or
            epoch_num == self.total_epochs or
            epoch_num % self.print_interval == 0
        )

        if should_print:
            metrics_str = f"Epoch {epoch_num}/{self.total_epochs} - {epoch_time:.1f}s"
            # ... rest of existing metrics ...
            print(f"[TIME_DEBUG] {metrics_str}")  # Add TIME_DEBUG prefix
```

### 6. Image Loading Performance

**File:** `src/data/image_processing.py:319` (load_and_preprocess_image function)

```python
def load_and_preprocess_image(filepath, bb_data, modality, target_size=(224, 224), augment=False):
    # ADD: Sample timing every 100 images
    import random
    should_time = random.random() < 0.01  # Time 1% of images

    if should_time:
        t_start = time.time()

    try:
        # Load image and convert to array
        img = load_img(filepath, color_mode="rgb", target_size=None)

        if should_time:
            t_after_load = time.time()
            print(f"[TIME_DEBUG] Image load (PIL): {(t_after_load - t_start)*1000:.1f}ms", flush=True)

        img_array = img_to_array(img)

        # ... rest of preprocessing ...

        if should_time:
            t_end = time.time()
            print(f"[TIME_DEBUG] Image preprocess total: {(t_end - t_start)*1000:.1f}ms", flush=True)

        return preprocessed_image
```

---

## Priority Action Plan

### Phase 1: Add Instrumentation (DO NOW)
1. Add all `[TIME_DEBUG]` prints from section above
2. Run a test with 1 fold to identify exact bottlenecks
3. Parse logs to see where time is actually spent
4. **Estimated time:** 2-3 hours (including 1 test run)

### Phase 2: Quick Wins (AFTER PHASE 1)
1. Increase `GLOBAL_BATCH_SIZE` to 128 (5 min change, test 1 epoch)
2. Cache RandomForest predictions (30 min change)
3. Fix pre-training cache logging (15 min change)
4. **Estimated speedup:** 1.5-2x
5. **Estimated time:** 1-2 hours (including testing)

### Phase 3: Major Optimization (AFTER PHASE 2)
1. Replace PIL image loading with `tf.io` (2-4 hours)
2. Move augmentation to GPU using `tf.image` (1-2 hours)
3. Test on single fold
4. **Estimated speedup:** 3-5x on top of Phase 2
5. **Estimated time:** 1 day (including testing and debugging)

### Phase 4: Polish (OPTIONAL)
1. Remove `[TIME_DEBUG]` prints
2. Add proper performance monitoring
3. Document optimal settings
4. **Estimated time:** 2-3 hours

---

## Testing Strategy

After each optimization:

1. **Single fold test (quick validation):**
   ```bash
   python src/main.py --data_percentage 30 --device-mode multi --verbosity 2 --fold 1
   ```
   - Should complete in ~15-30 min (baseline) → ~3-8 min (optimized)

2. **Full test (3 folds):**
   ```bash
   python agent_communication/generative_augmentation/test_generative_aug.py
   ```
   - Baseline test should go from 442 min → ~55-110 min

3. **Measure success:**
   - Parse `[TIME_DEBUG]` logs to confirm bottleneck is reduced
   - Check nvidia-smi during training (GPU compute should be 70-95%)
   - Verify accuracy doesn't degrade

---

## Files to Modify (Summary)

| Priority | File | Lines | Change |
|----------|------|-------|--------|
| **Critical** | `src/data/dataset_utils.py` | 234-274 | Replace PIL with tf.io |
| **Critical** | `src/data/generative_augmentation_v3.py` | create_enhanced_augmentation_fn | GPU-based augmentation |
| **Critical** | `src/utils/production_config.py` | 29 | Increase batch size to 128 |
| **High** | `src/data/dataset_utils.py` | 1212-1234 | Cache RF predictions |
| **High** | `src/training/training_utils.py` | 1274-1283 | Fix pretrain cache logging |
| **Medium** | `src/training/training_utils.py` | 99-137 | Add epoch timing |
| **Low** | All above files | Various | Add [TIME_DEBUG] instrumentation |

---

## Expected Results

**Current (Baseline Test 1):**
- Total: 442 minutes
- Per epoch: 69-142 seconds
- GPU utilization: <30%

**After Phase 2 (Quick Wins):**
- Total: ~220-300 minutes (1.5-2x faster)
- Per epoch: 40-80 seconds
- GPU utilization: 40-50%

**After Phase 3 (Major Optimizations):**
- Total: ~55-110 minutes (4-8x faster)
- Per epoch: 8-20 seconds
- GPU utilization: 70-95%

---

## Notes

- All timing is for Test 1 (baseline, NO SDXL)
- Test 2 (with SDXL) will benefit from same optimizations
- SDXL generation itself is already GPU-accelerated (correct)
- Focus is on baseline training pipeline, not SDXL generation
- Pre-training cache issue is worth fixing for interrupted runs, but not critical for normal runs (each fold needs different weights)

---
