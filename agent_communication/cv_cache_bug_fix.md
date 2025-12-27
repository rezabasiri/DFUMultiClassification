# CRITICAL BUG FIX: TensorFlow Cache Reuse Across CV Folds

## Date
2025-12-27

## Severity
🔴 **CRITICAL** - Completely invalidates k-fold cross-validation results

## The Bug

### Problem
When using k-fold cross-validation with `--cv_folds > 1`, **all folds were validating on the same data** (Fold 1's validation set) due to TensorFlow dataset cache reuse.

### Root Cause

**File**: [src/data/dataset_utils.py:346-349](../src/data/dataset_utils.py#L346-L349)

**Original Code**:
```python
modality_suffix = '_'.join(sorted(selected_modalities))
cache_filename = f'tf_cache_train_{modality_suffix}' if is_training else f'tf_cache_valid_{modality_suffix}'
```

Cache filename was ONLY based on:
- Split type (train/valid)
- Modality combination

**Missing**: Fold/run identifier!

### How It Failed

With 3-fold CV and metadata modality:

**Fold 1** (iteration_idx=0, run=0):
- Creates `tf_cache_valid_metadata` with Patient 5's data ✅
- Validates on correct patients
- Tracks misclassifications for Patient 5

**Fold 2** (iteration_idx=1, run=1):
- Tries to create `tf_cache_valid_metadata`
- **TensorFlow finds existing cache from Fold 1** 📁
- **Reuses Fold 1's validation data instead of Fold 2's!** ❌
- Validates on WRONG patients (Fold 1's patients, including Patient 5)
- Tracks misclassifications for Patient 5 AGAIN

**Fold 3** (iteration_idx=2, run=2):
- Same as Fold 2 - reuses Fold 1's cache ❌
- Validates on Fold 1's patients AGAIN
- Tracks misclassifications for Patient 5 a THIRD time

### Evidence

**Symptom**: All misclassified samples have `Misclass_Count=3` in the CSV
```csv
Patient,Appointment,DFU,True_Label,Sample_ID,Misclass_Count
2,0,1,I,P002A00D1,3
5,3,1,P,P005A03D1,3
10,3,1,R,P010A03D1,3
```

With proper patient-level CV, we should see varied counts (0-3), since each patient appears in validation only once per run.

**Test Verification**:
```python
# create_patient_folds() correctly assigns each patient to ONE fold
Patient 5 appears in validation in fold(s): [2]  # Only Fold 2 ✅
```

But CSV shows Patient 5 with count=3, meaning it was misclassified in ALL 3 folds! ❌

### Cache Files Created
```bash
$ ls -lh results/tf_records/
tf_cache_valid_metadata.data-00000-of-00001  # SAME file used by all folds!
tf_cache_valid_metadata.index
```

## The Fix

### TWO Root Causes Identified

The bug had **TWO independent root causes** (either alone would cause the bug):

1. ❌ **Cache filename didn't include fold ID** → Folds 2 & 3 would reuse Fold 1's cache
2. ❌ **`clear_cache_files()` looked in wrong directory** → Cache never deleted between folds

**Both needed to be fixed!**

### Modified Files

**1. [src/data/dataset_utils.py:183-197](../src/data/dataset_utils.py#L183-L197)**

**Added `fold_id` parameter** to `create_cached_dataset()`:
```python
def create_cached_dataset(best_matching_df, selected_modalities, batch_size,
                         is_training=True, cache_dir=None, augmentation_fn=None,
                         image_size=128, fold_id=0):  # ← NEW PARAMETER
```

2. **Include `fold_id` in cache filename** (lines 345-349):
```python
# Cache the dataset with unique filename per modality combination AND fold
# CRITICAL: Include fold_id to prevent cache reuse across different CV folds
modality_suffix = '_'.join(sorted(selected_modalities))
split_type = 'train' if is_training else 'valid'
cache_filename = f'tf_cache_{split_type}_{modality_suffix}_fold{fold_id}'  # ← fold_id added
```

3. **Pass `run` from `prepare_cached_datasets`** (lines 1011, 1021):
```python
train_dataset, pre_aug_dataset, steps_per_epoch = create_cached_dataset(
    train_data,
    selected_modalities,
    batch_size,
    is_training=True,
    cache_dir=cache_dir,
    augmentation_fn=create_enhanced_augmentation_fn(gen_manager, aug_config) if gen_manager else None,
    image_size=image_size,
    fold_id=run)  # ← Pass fold/run ID

valid_dataset, _, validation_steps = create_cached_dataset(
    valid_data,
    selected_modalities,
    batch_size,
    is_training=False,
    cache_dir=cache_dir,
    augmentation_fn=None,
    image_size=image_size,
    fold_id=run)  # ← Pass fold/run ID
```

**2. [src/training/training_utils.py:1897-1921](../src/training/training_utils.py#L1897-L1921)**

**Fixed `clear_cache_files()` to look in correct directory**:
```python
def clear_cache_files():
    """Clear any existing cache files."""
    import glob

    # FIXED: Include tf_records subdirectory where cache files are actually stored
    cache_patterns = [
        os.path.join(result_dir, 'tf_records', 'tf_cache_train*'),  # ← ADDED
        os.path.join(result_dir, 'tf_records', 'tf_cache_valid*'),  # ← ADDED
        os.path.join(result_dir, 'tf_cache_train*'),  # Legacy location
        os.path.join(result_dir, 'tf_cache_valid*'),  # Legacy location
        'tf_cache_train*',
        'tf_cache_valid*'
    ]
```

**Before**: Looked in `results/tf_cache_*` (wrong!)
**After**: Looks in `results/tf_records/tf_cache_*` (correct!)

### After Fix

Cache files will now be:
```bash
tf_cache_valid_metadata_fold0.data-00000-of-00001  # Fold 1's validation data
tf_cache_valid_metadata_fold0.index
tf_cache_valid_metadata_fold1.data-00000-of-00001  # Fold 2's validation data
tf_cache_valid_metadata_fold1.index
tf_cache_valid_metadata_fold2.data-00000-of-00001  # Fold 3's validation data
tf_cache_valid_metadata_fold2.index
```

Each fold now has its own unique cache! ✅

## Impact

### What Was Broken

**ALL k-fold CV results were invalid:**
- ❌ Reported metrics (accuracy, F1, etc.) were calculated on the same validation set 3 times
- ❌ Misclassification tracking counted the same samples multiple times
- ❌ Dataset polishing thresholds were based on inflated counts
- ❌ Phase 2 optimization used contaminated data

### What This Affects

1. **Bayesian optimization results** - All Phase 1 metrics were wrong
2. **Dataset filtering** - Samples excluded based on inflated misclass counts
3. **Model evaluation** - Reported performance metrics were not true k-fold CV
4. **Research validity** - Any published results using k-fold CV need re-evaluation

### Why This Wasn't Caught Earlier

1. **Patient-level splitting looked correct** when tested independently
2. **Cache is invisible** - TensorFlow silently reuses existing cache
3. **Metrics looked reasonable** - similar to single-run results
4. **No warnings** - TensorFlow doesn't warn about cache reuse

## Testing the Fix

### Before Fix
```bash
# Delete old cache
rm -rf /workspace/DFUMultiClassification/results/tf_records/*

# Run 3-fold CV
python src/main.py --mode search --cv_folds 3 --selected_modalities metadata

# Check misclass counts - ALL will be 3
awk -F',' 'NR>1 {print $6}' results/misclassifications/frequent_misclassifications_metadata.csv | sort -u
# Output: 3 (ALL samples counted 3 times)
```

### After Fix
```bash
# Delete old cache
rm -rf /workspace/DFUMultiClassification/results/tf_records/*
rm -rf /workspace/DFUMultiClassification/results/misclassifications/*

# Run 3-fold CV
python src/main.py --mode search --cv_folds 3 --selected_modalities metadata

# Check misclass counts - Should vary from 1-3
awk -F',' 'NR>1 {print $6}' results/misclassifications/frequent_misclassifications_metadata.csv | sort | uniq -c
# Output:
#   XX 1  (misclassified in 1 fold only)
#   XX 2  (misclassified in 2 folds)
#   XX 3  (misclassified in all 3 folds)
```

### Verify Unique Cache Files
```bash
ls -1 results/tf_records/ | grep metadata
# Should see:
# tf_cache_train_metadata_fold0.data-00000-of-00001
# tf_cache_train_metadata_fold0.index
# tf_cache_train_metadata_fold1.data-00000-of-00001
# tf_cache_train_metadata_fold1.index
# tf_cache_train_metadata_fold2.data-00000-of-00001
# tf_cache_train_metadata_fold2.index
# tf_cache_valid_metadata_fold0.data-00000-of-00001
# tf_cache_valid_metadata_fold0.index
# tf_cache_valid_metadata_fold1.data-00000-of-00001
# tf_cache_valid_metadata_fold1.index
# tf_cache_valid_metadata_fold2.data-00000-of-00001
# tf_cache_valid_metadata_fold2.index
```

## Recommendations

### Immediate Actions
1. ✅ **Fix applied** - Cache now includes fold ID
2. 🔴 **Delete all existing misclassification CSVs** - They contain inflated counts
3. 🔴 **Delete Bayesian optimization results** - Based on invalid metrics
4. 🔴 **Re-run Phase 1 detection** with fixed code
5. 🔴 **Re-run Phase 2 optimization** with corrected data

### Re-run Command
```bash
# Clear ALL previous results
rm -rf /workspace/DFUMultiClassification/results/tf_records/*
rm -rf /workspace/DFUMultiClassification/results/misclassifications/*
rm -f /workspace/DFUMultiClassification/results/misclassifications_saved/bayesian_optimization_results.json

# Re-run auto_polish from scratch
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 7 \
  --phase2-cv-folds 7 \
  --phase1-n-runs 1 \
  --n-evaluations 30 \
  --device-mode multi \
  --min-minority-retention 0.85 \
  --track-misclass valid
```

## Related Issues

This fix also resolves the mystery of "identical metadata producing different predictions":
- The predictions WERE identical within each fold
- But we were seeing accumulated results from the SAME fold being validated 3 times
- Different batches/runs of the same fold data produced slightly different predictions due to numerical instability
- These got tracked as separate entries in the old buggy misclassification tracking

## Lessons Learned

1. **Always include iteration identifiers in cache keys** when doing k-fold CV
2. **Test cache behavior explicitly** - don't assume frameworks handle it correctly
3. **Validate CV split integrity** - check that validation sets are actually different across folds
4. **Monitor cache file creation** - suspicious when only ONE cache file exists for multiple folds

## Credit

Bug discovered through detailed investigation of misclassification counts being uniformly equal to the number of CV folds.
