# Phase 1 Multiple Runs Fix

## Problem

When using `--phase1-n-runs > 1` with `--phase1-cv-folds > 1`, all runs were producing **identical patient fold splits**, causing:

1. **Duplicate misclassification tracking**: Same samples misclassified in same way across runs
2. **Inflated counts**: Misclass_Count artificially multiplied by number of runs
3. **Wasted computation**: Multiple runs added no value since results were identical

### Root Cause

`create_patient_folds()` always used `random_state=42` for fold generation. Even though auto_polish deleted `patient_split_*.npz` files between runs, the splits regenerated identically.

## Solution

Added `CV_FOLD_SEED` environment variable to allow different fold splits per run:

### Changes Made

1. **[src/data/dataset_utils.py:46-50](../../src/data/dataset_utils.py#L46-L50)**
   ```python
   # Allow override via environment variable
   if 'CV_FOLD_SEED' in os.environ:
       try:
           random_state = int(os.environ['CV_FOLD_SEED'])
       except ValueError:
           pass  # Keep default if invalid
   ```

2. **[scripts/auto_polish_dataset_v2.py:868-875](../../scripts/auto_polish_dataset_v2.py#L868-L875)**
   ```python
   if self.phase1_cv_folds <= 1:
       # Single-split mode: set seed for data splitting
       os.environ['CROSS_VAL_RANDOM_SEED'] = str(self.base_random_seed + run_counter)
   else:
       # CV mode: set seed for fold generation to get different splits each run
       os.environ['CV_FOLD_SEED'] = str(self.base_random_seed + run_counter)
   ```

## Behavior Now

### With `--phase1-cv-folds 7 --phase1-n-runs 3`:

- **Run 1**: Folds generated with seed=43 → 7-fold CV → Collect misclassifications
- **Run 2**: Folds generated with seed=44 → **DIFFERENT** patient splits → New misclassifications
- **Run 3**: Folds generated with seed=45 → **DIFFERENT** patient splits → New misclassifications

Each sample can now be misclassified 0-21 times (3 runs × 7 folds max), providing robust frequency detection.

## Recommendations

### For Phase 1 Misclassification Detection:

**Option 1: High confidence (slower)**
```bash
--phase1-cv-folds 5 --phase1-n-runs 3
```
- 15 total evaluations per sample
- More robust detection of problematic samples
- Better for critical datasets

**Option 2: Balanced (recommended)**
```bash
--phase1-cv-folds 7 --phase1-n-runs 1
```
- 7 evaluations per sample (1 per fold)
- Fast, still provides good coverage
- Each sample validated once with different training contexts

**Option 3: Quick screening**
```bash
--phase1-cv-folds 3 --phase1-n-runs 1
```
- 3 evaluations per sample
- Fastest option for initial exploration

### Retention Settings

With proper multiple runs, you can use more aggressive filtering:

- **Conservative**: `--min-minority-retention 0.70` (keep 70% of minority class)
- **Moderate**: `--min-minority-retention 0.50` (keep 50% of minority class)
- **Aggressive**: `--min-minority-retention 0.30` (keep 30% of minority class)

The retention is calculated at **sample level** (unique patient-appointment-DFU combinations), which indirectly affects total image count.
