# ⚠️ TEMPORARY COMPARISON MODE ACTIVE

## Current Status

**Two temporary modifications active** for comparison testing:

1. Smart alpha capping is DISABLED
2. Force fresh training parameter added to original code

### What Changed

#### 1. Alpha Capping (dataset_utils.py)

**File**: `src/data/dataset_utils.py` (lines 611-652)

**Before** (smart capping):
```python
# Normalize to sum=1.0 first
alpha_sum = sum(alpha_values)
alpha_values = [alpha/alpha_sum for alpha in alpha_values]

# Cap and redistribute at MAX_ALPHA=0.5
# ... (smart redistribution algorithm)
```

**Now** (original normalization):
```python
# TEMPORARY: Use original normalization for comparison testing
alpha_sum = sum(alpha_values)
alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]
```

This matches `main_original.py` exactly.

#### 2. Force Fresh Training (main_original.py)

**File**: `src/main_original.py` (lines 2695, 3028)

**Before**:
```python
def cross_validation_manual_split(data, configs, train_patient_percentage=0.8, n_runs=3):
    # ...
    if os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):
        model.load_weights(...)
```

**Now**:
```python
def cross_validation_manual_split(data, configs, train_patient_percentage=0.8, n_runs=3, force_fresh=False):
    # ...
    # TEMPORARY MODIFICATION FOR COMPARISON TESTING
    # To revert: remove 'not force_fresh and' from the condition below
    # Original code:
    # if os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):
    if not force_fresh and os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):
        model.load_weights(...)
```

This allows the comparison script to force fresh training by passing `force_fresh=True`, preventing cached weight loading.

## Why

To do an **apples-to-apples** comparison:
- Same alpha calculation
- Same model architecture
- Same data preprocessing

Any metric differences will be due to **code refactoring only**, not algorithmic improvements.

## Expected Results

With identical alpha values, the refactored code should produce:
- ✅ **Identical or near-identical metrics** (< 0.01% difference)
- ✅ **Same alpha values** (e.g., `[0.748, 0.282, 1.971]`)
- ✅ **Same training behavior**

## After Testing

**IMPORTANT**: Restore both modifications after comparison is complete!

### Revert Instructions

**Option 1: Git Revert** (recommended)
```bash
# Revert the comparison testing commits
git revert HEAD~2..HEAD  # Adjust range as needed
```

**Option 2: Manual Revert**

1. **Restore smart alpha capping**:
   - File: `src/data/dataset_utils.py` lines 611-652
   - Uncomment the smart capping code
   - Comment out the temporary normalization

2. **Remove force_fresh parameter**:
   - File: `src/main_original.py` line 2695
   - Change: `def cross_validation_manual_split(data, configs, train_patient_percentage=0.8, n_runs=3):`
   - File: `src/main_original.py` line 3028
   - Change: `if os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):`
   - Remove the TEMPORARY MODIFICATION comments

## Comparison Command

```bash
# Run comparison with identical alpha values
python scripts/simple_comparison_test.py metadata

# Should see matching alpha values in both versions:
# Alpha values (ordered) [I, P, R]: [0.748, 0.282, 1.971]

# Keep files for inspection (skip automatic cleanup)
python scripts/simple_comparison_test.py metadata --no-cleanup

# Test with more data
python scripts/simple_comparison_test.py metadata --data_pct 50
```

## File Management

**Automatic Cleanup** (default):
- Comparison files are automatically deleted after test completes
- Keeps repository clean
- No manual cleanup needed

**Manual Control**:
```bash
# Keep files after test
python scripts/simple_comparison_test.py metadata --no-cleanup

# Manual cleanup later
python -c "from src.utils.config import cleanup_for_resume_mode; cleanup_for_resume_mode('fresh')"
```

**What Gets Created**:
- Model weights: `results/models/*.h5`
- Predictions: `results/checkpoints/*pred*.npy`
- Patient splits: `results/checkpoints/patient_split*.npz`
- CSV results: `results/csv/*.csv`
- TF cache: `results/tf_records/`

**Gitignore Coverage**: ✅ All comparison files are automatically ignored by `.gitignore`

---

**Created**: 2025-12-21
**Purpose**: Controlled comparison testing
**Status**: ⚠️ TEMPORARY - REVERT AFTER TESTING
