# Thread Exhaustion Fix - All Changes

**Issue**: TensorFlow crashes with `pthread_create() failed (EAGAIN)` during Fold 3/3 of 3-fold cross-validation on 5 GPUs. Docker container has a cgroup PIDs limit of 7680.

**Branch**: `claude/optimize-preprocessing-speed-0dVA4`

---

## 1. OMP_NUM_THREADS Before TF Import

**File**: `src/main.py` (lines 19-21)

**Change**: Added `OMP_NUM_THREADS`, `TF_NUM_INTEROP_THREADS`, `TF_NUM_INTRAOP_THREADS` as environment variables BEFORE `import tensorflow`.

**Before**:
```python
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# (threading was set at line 2423 via apply_environment_config(), AFTER TF import at line 42)
```

**After**:
```python
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
```

**Rationale**: OpenMP reads `OMP_NUM_THREADS` at library load time (when TF's shared library is loaded during import). Setting it after import has no effect. On a 32-core machine, this was causing 32 OpenMP threads per pool instead of 2. Measured: 63 threads after import without fix vs 3 threads with fix.

**Revert risk**: LOW - This is a genuine bug fix. The values were already configured in `production_config.py` but applied too late.

---

## 2. AUTOTUNE Replaced in `dataset_utils.py`

**File**: `src/data/dataset_utils.py`

**Changes** (3 locations):
- Line ~362: `num_parallel_calls=tf.data.AUTOTUNE` → `num_parallel_calls=4` (preprocessing map)
- Line ~408: `num_parallel_calls=tf.data.AUTOTUNE` → `num_parallel_calls=4` (augmentation map)
- Line ~420: `dataset.prefetch(tf.data.AUTOTUNE)` → `dataset.prefetch(2)`

**Rationale**: `AUTOTUNE` can spawn up to `nproc` (32) threads per map operation. With multiple datasets per fold and 5 GPU replicas, this creates hundreds of threads. Fixed values of 4 (map) and 2 (prefetch) are sufficient for the data pipeline.

**Revert risk**: MEDIUM - Reverting may slightly improve data loading throughput on CPU but risks thread exhaustion. The values 4 and 2 should be more than adequate for the pipeline given GPU-bound training.

---

## 3. AUTOTUNE Replaced in `caching.py`

**File**: `src/data/caching.py`

**Changes** (3 locations):
- Line ~173: `num_parallel_calls=tf.data.AUTOTUNE` → `num_parallel_calls=4` (preprocessing map)
- Line ~202: `num_parallel_calls=tf.data.AUTOTUNE` → `num_parallel_calls=4` (augmentation map)
- Line ~212: `dataset.prefetch(tf.data.AUTOTUNE)` → `dataset.prefetch(2)`

**Rationale**: Same as dataset_utils.py. This file is the older dataset creation path; may not be actively used but changed for consistency.

**Revert risk**: LOW - This file may not be in the active code path. Safe to revert if caching.py is not used.

---

## 4. AUTOTUNE Replaced in `training_utils.py` Map Calls

**File**: `src/training/training_utils.py`

**Changes** (4 locations):
- Line ~1101: `train_dataset.map(remove_sample_id_for_training, num_parallel_calls=tf.data.AUTOTUNE)` → `num_parallel_calls=4`
- Line ~1102: `valid_dataset.map(remove_sample_id_for_training, num_parallel_calls=tf.data.AUTOTUNE)` → `num_parallel_calls=4`
- Line ~1246: `pretrain_train_dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)` → `num_parallel_calls=4`
- Line ~1248: `pretrain_valid_dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)` → `num_parallel_calls=4`

**Rationale**: The `remove_sample_id_for_training` function is trivial (just removes a dict key) — it does not need 32 parallel threads. 4 is more than sufficient.

**Revert risk**: LOW - These map operations are lightweight dict filtering. No performance benefit from AUTOTUNE here.

---

## 5. AUTOTUNE Replaced in `main.py` Gating Network Datasets

**File**: `src/main.py` (lines ~906, ~910)

**Changes** (2 locations):
- `dataset.prefetch(tf.data.AUTOTUNE)` → `dataset.prefetch(2)` (gating train dataset)
- `val_dataset.prefetch(tf.data.AUTOTUNE)` → `val_dataset.prefetch(2)` (gating validation dataset)

**Rationale**: The gating network datasets are small (in-memory tensor slices, not image loading). AUTOTUNE prefetch is overkill.

**Revert risk**: LOW - Negligible performance difference for small in-memory datasets.

---

## 6. `private_threadpool_size` on Distributed Datasets

**File**: `src/training/training_utils.py`

**Changes** (2 locations):

**Pre-training datasets** (line ~1250):
```python
_thread_opts = tf.data.Options()
_thread_opts.threading.private_threadpool_size = 8
pretrain_train_dataset = pretrain_train_dataset.with_options(_thread_opts)
pretrain_valid_dataset = pretrain_valid_dataset.with_options(_thread_opts)
```

**Main training datasets** (line ~1383):
```python
_thread_opts = tf.data.Options()
_thread_opts.threading.private_threadpool_size = 8
train_dataset_dis = strategy.experimental_distribute_dataset(train_dataset.with_options(_thread_opts))
valid_dataset_dis = strategy.experimental_distribute_dataset(valid_dataset.with_options(_thread_opts))
```

**Rationale**: When datasets are distributed across 5 GPUs via `experimental_distribute_dataset()`, each replica can create its own thread pool. `private_threadpool_size=8` caps this per-dataset to prevent unbounded growth.

**Revert risk**: MEDIUM - Removes the per-dataset thread cap. If other fixes are sufficient, this may be unnecessary, but it's a safety net.

---

## 7. Pre-training Source Dataset Cleanup

**File**: `src/training/training_utils.py` (line ~1348)

**Before**:
```python
del pretrain_model  # Free memory
```

**After**:
```python
del pretrain_model, pretrain_train_dis, pretrain_valid_dis, pretrain_train_dataset, pretrain_valid_dataset
```

**Rationale**: The original code only deleted the model but left the pre-training datasets alive. These hold map/prefetch thread pools. Now all 5 pre-training objects are deleted together.

**Revert risk**: LOW - This is a straightforward resource cleanup. No reason to revert.

---

## 8. Comprehensive End-of-Fold Cleanup

**File**: `src/training/training_utils.py` (lines ~1814-1846)

**Before** (original code before any changes):
```python
tf.keras.backend.clear_session()
gc.collect()
clear_gpu_memory()
```

**After**:
```python
model = None
train_dataset_dis = None
valid_dataset_dis = None
train_dataset = None
valid_dataset = None
master_train_dataset = None
master_valid_dataset = None
pre_aug_dataset = None          # NEW - was leaked
pre_aug_train_dataset = None    # NEW - was leaked
valid_dataset_with_ids = None   # NEW - was leaked
data_manager = None             # NEW - was leaked
aug_config = None               # NEW - was leaked
tf.keras.backend.clear_session()
gc.collect()
gc.collect()                    # NEW - second pass for reference cycles
clear_gpu_memory()
```

**Rationale**: Multiple per-fold objects were never cleaned up between folds:
- `pre_aug_dataset` — created at line 1027, used for prediction tracking
- `pre_aug_train_dataset` — created at line 1091 per config iteration
- `valid_dataset_with_ids` — created at line 1657 per config iteration
- `data_manager` — ProcessedDataManager created at line 983
- `aug_config` — AugmentationConfig created at line 989

Each holds references to datasets with active thread pools. Setting to None + double gc.collect releases these.

**Revert risk**: LOW - This is correct resource cleanup. The leaked objects have no purpose after the fold completes.

---

## 9. Comprehensive Thread Debug Logging (TEMPORARY - remove after issue resolved)

Added `_log_threads()` helper functions and `[THREADS]` log lines at every key point in the training pipeline to identify exactly where threads accumulate. All output lines are prefixed with `[THREADS]` for easy grepping.

**To find all debug output**: `grep '\[THREADS\]' <log_file>`

**To remove all debug code**: Search for `_log_threads` and the comment `--- Thread debug helper ---` in each file.

### File: `src/main.py`

Added `_log_threads_main()` helper function (after line ~47) and log points at:

| Location | Label | What it measures |
|----------|-------|------------------|
| Before `import tensorflow` | `before TF import` | Baseline thread count |
| After `import tensorflow` | `after TF import` | Threads created by TF library loading (OpenMP pools) |
| After `setup_device_strategy()` | `after strategy creation` | Threads from MirroredStrategy + NCCL + CUDA |
| Before gating `model.fit()` | `before gating fit` | Threads before gating network training |
| After gating `model.fit()` | `after gating fit` | Threads created by gating training |

### File: `src/training/training_utils.py`

Added `_log_threads()` helper function (after line ~64) and log points at:

| Location | Label | What it measures |
|----------|-------|------------------|
| Start of fold loop | `Fold X/Y START` | Baseline for each fold (should be stable) |
| After `prepare_cached_datasets()` | `Fold X/Y after prepare_cached_datasets` | Threads from dataset creation + RF models |
| After `create_multimodal_model()` | `Fold X/Y after model creation` | Threads from model building in strategy scope |
| After pre-training `model.fit()` | `Fold X/Y after pretrain fit` | Threads from depth_rgb pre-training |
| After pre-training `del` cleanup | `Fold X/Y after pretrain cleanup` | Whether pretrain threads are released |
| After `compile()` + `distribute_dataset()` | `Fold X/Y after compile+distribute` | Threads from distributing main datasets |
| After Stage 1 `model.fit()` | `Fold X/Y after Stage 1 fit` | Threads from frozen-branch training |
| After Stage 2 `model.fit()` | `Fold X/Y after Stage 2 fit` | **Crash point** - threads from fine-tuning |
| After all training (single or 2-stage) | `Fold X/Y after all training` | Total training threads |
| Before gating network | `Fold X/Y before gating network` | Threads before gating (after evaluation) |
| After end-of-fold cleanup | `Fold X/Y AFTER CLEANUP` | **Key metric** - should be stable across folds |

### File: `src/data/dataset_utils.py`

Added `_log_threads_ds()` helper function (after line ~20) and log points at:

| Location | Label | What it measures |
|----------|-------|------------------|
| Before RF `fit()` | `before RF fit (n_jobs=-1)` | Baseline before sklearn spawns threads |
| After RF `fit()` | `after RF fit (n_jobs=-1)` | Threads created by joblib (n_jobs=-1 = 32 cores) |
| End of `prepare_cached_datasets()` | `end of prepare_cached_datasets` | Total after all dataset/RF creation |

**Revert**: Remove the `_log_threads*` functions and all `_log_threads*()` calls from the 3 files. Search for `# --- Thread debug helper` to find the function blocks, and `_log_threads` to find all call sites.

---

## 10. Subprocess Isolation Per Fold (THE FIX)

**Files**: `src/main.py`, `src/training/training_utils.py`, `agent_communication/generative_augmentation/test_generative_aug.py`

**Root cause analysis**: After running the full training pipeline with debug logging (change #9), thread/PID counts were found to be **stable across folds** (not accumulating). Cleanup works correctly. However, TF/CUDA accumulates invisible C++ resources (memory maps, internal buffers) that can't be freed within the same process. Simulations confirmed:
- Memory maps grow ~730/fold and never get freed by `clear_session()`
- RSS grows ~550MB/fold even after cleanup
- `vm.max_map_count=65530` is read-only in Docker

**Solution**: Run each fold in a separate subprocess via `--fold N` argument.

### Changes:

**main.py**:
- Added `--fold N` CLI argument (1-indexed, default=None for all folds)
- Passed `target_fold` through `main()` → `main_search()` → `cross_validation_manual_split()`

**training_utils.py**:
- Added `target_fold` parameter to `cross_validation_manual_split()`
- When `target_fold` is set, non-target folds are skipped and their metrics loaded from disk
- Modified `is_run_complete()` skip to also load saved metrics (so aggregated results work across subprocess runs)
- Added `load_run_metrics()` function to read `modality_results_run_{N}.csv`

**test_generative_aug.py**:
- Modified `run_test()` to run 3 separate subprocess calls (one per fold) instead of one
- First fold: `--resume_mode fresh --fold 1` (clears old checkpoints)
- Subsequent folds: `--resume_mode auto --fold N` (preserves previous results)
- Each subprocess gets a fresh TF/CUDA state, preventing resource accumulation

**Usage**:
```bash
# Per-fold execution (new):
python src/main.py --fold 1 --resume_mode fresh --device-mode multi
python src/main.py --fold 2 --device-mode multi
python src/main.py --fold 3 --device-mode multi

# All folds in one process (original, still works):
python src/main.py --device-mode multi
```

**Revert risk**: LOW - The `--fold` argument is optional. Without it, behavior is unchanged. The test script change can be reverted by restoring the single-subprocess run_test() function.

---

## 11. Enhanced Debug Logging (memory maps + RSS)

**Files**: `src/main.py`, `src/training/training_utils.py`, `src/data/dataset_utils.py`

**Change**: All `_log_threads*()` helper functions now also report:
- `maps=N/65530`: Count of `/proc/self/maps` entries (vm.max_map_count limit)
- `rss=NMB`: Process resident set size in MB

**Example output**:
```
[THREADS] Fold 1/3 START: proc=36, cgroup=506/7680, maps=1850/65530, rss=1200MB
```

**Rationale**: Thread/PID counts proved insufficient to diagnose the crash. Memory maps and RSS track TF/CUDA internal resource accumulation that isn't visible in thread counts.

**Revert risk**: NONE - Diagnostic only. Remove with other debug logging (change #9).

---

## Summary Table

| # | File | Change | Revert Risk | Category |
|---|------|--------|-------------|----------|
| 1 | main.py | OMP_NUM_THREADS before import | LOW | Bug fix |
| 2 | dataset_utils.py | AUTOTUNE → fixed (4/2) | MEDIUM | Thread reduction |
| 3 | caching.py | AUTOTUNE → fixed (4/2) | LOW | Thread reduction |
| 4 | training_utils.py | AUTOTUNE → 4 in map calls | LOW | Thread reduction |
| 5 | main.py | AUTOTUNE → 2 in gating prefetch | LOW | Thread reduction |
| 6 | training_utils.py | private_threadpool_size=8 | MEDIUM | Thread cap |
| 7 | training_utils.py | Delete pretrain source datasets | LOW | Resource cleanup |
| 8 | training_utils.py | Full end-of-fold cleanup | LOW | Resource cleanup |
| 9 | main.py, training_utils.py, dataset_utils.py | Thread debug logging | NONE | Diagnostics |
| 10 | main.py, training_utils.py, test_generative_aug.py | **Subprocess isolation per fold** | **LOW** | **THE FIX** |
| 11 | main.py, training_utils.py, dataset_utils.py | Enhanced debug (maps+RSS) | NONE | Diagnostics |

**Candidates to revert first** (if issue is resolved and you want AUTOTUNE back): #2 and #6. These are the ones most likely to affect data loading performance. Monitor thread counts via #9 after reverting to ensure safety.

**Should NOT revert**: #1 (genuine bug), #7 and #8 (correct cleanup), #4 and #5 (trivial operations that don't benefit from AUTOTUNE), #10 (subprocess isolation is the actual fix).

**Remove after issue resolved**: #9 and #11 (all debug logging). Search for `_log_threads` across the codebase.

---

## Key Info for Diagnosis

- **Docker cgroup PIDs limit**: 7680 (read-only, cannot increase)
- **vm.max_map_count**: 65530 (read-only, cannot increase)
- **System idle PIDs**: ~460
- **Crash signature**: `Check failed: ret == 0 (11 vs. 0) Thread tf_ creation via pthread_create() failed.`
- **Crash location**: Consistently at Fold 3/3, during Stage 2 training (~7-8 epochs in)
- **Key finding**: Thread/PID counts are STABLE across folds (~590 cgroup at crash point, same as Fold 2). The crash is caused by invisible TF/CUDA C++ resource accumulation (memory maps, internal buffers)
- **What to look for in logs**: `[THREADS]` lines with `maps=` values growing significantly across folds
